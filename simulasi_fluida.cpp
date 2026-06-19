#define CL_HPP_TARGET_OPENCL_VERSION 200 // Gunakan standar C++ OpenCL 2.0
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <optional>

#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include <omp.h> // Header wajib OpenMP
#include <chrono> // Untuk mengukur waktu presisi tinggi jika diperlukan

// Fungsi helper membaca file .cl
std::string readKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

using HRClock = std::chrono::high_resolution_clock;

// ==============================================================================
// DEKLARASI FUNGSI
// ==============================================================================
void updateGPU (cl::CommandQueue& queue, 
                cl::Kernel& kernelKecepatan, cl::Kernel& kernelPosisi, 
                cl::Buffer& bufferPos, cl::Buffer& bufferVel,
                cl::Buffer& bufferCellStart, cl::Buffer& bufferCellCount,
                std::vector<sf::Vector2f>& positions, std::vector<sf::Vector2f>& velocities, const sf::FloatRect& container, 
                float cellSize, int gridWidth, int gridHeight, float deltaTime, float fluidStiffness,
                sf::Vector2f mousePos, float mouseForce, int numParticles, cl::Event* profEvent);
void render(sf::RenderWindow& window, const std::vector<sf::Vector2f>& positions, 
            const std::vector<sf::Color>& colors, const sf::FloatRect& container);
void initializeParticles(std::vector<sf::Vector2f>& positions, std::vector<sf::Color>& colors, 
                         const sf::FloatRect& container);
void mapParticlesToGrid(const std::vector<sf::Vector2f>& positions, 
                        const sf::FloatRect& container, 
                        float cellSize, 
                        int gridWidth, 
                        int gridHeight,
                        std::vector<int>& cellCount, 
                        std::vector<int>& particleCellIDs);
void buildGridOffsetsAndSort(const std::vector<sf::Vector2f>& positions,
                             const std::vector<sf::Vector2f>& velocities,
                             const std::vector<int>& particleCellIDs,
                             const std::vector<int>& cellCount,
                             std::vector<int>& cellStart,
                             std::vector<sf::Vector2f>& sortedPositions,
                             std::vector<sf::Vector2f>& sortedVelocities);
sf::Color getSpeedColor(float vx, float vy, float maxSpeed);

unsigned int windowWidth = 1800;
unsigned int windowHeight = 1100;

// ==============================================================================
// FUNGSI UTAMA (MAIN)
// ==============================================================================
int main() {
    // --- [1] SETUP KONTOINER ---
    sf::Vector2f containerSize(((float) windowWidth) - 100, ((float) windowHeight) - 100); 
    sf::Vector2f containerPos((windowWidth - containerSize.x) / 2.0f, (windowHeight - containerSize.y) / 2.0f);
    sf::FloatRect container(containerPos, containerSize);

    // --- [2] SETUP PARTIKEL (SoA - Structure of Arrays) ---
    int numParticles = 50000; // Coba langsung 50.000 partikel!
    float fluidStiffness = .01f;
    std::vector<sf::Vector2f> positions(numParticles);
    std::vector<sf::Color> colors(numParticles);
    std::vector<sf::Vector2f> velocities(numParticles, {0.0f, 0.0f});
    initializeParticles(positions, colors, container);

    const float cellSize = 100.0f; 
    int gridWidth  = static_cast<int>(container.size.x / cellSize); // 1100 / 20 = 55 kolom
    int gridHeight = static_cast<int>(container.size.y / cellSize); // 800 / 20  = 40 baris
    int totalCells = gridWidth * gridHeight; // 55 * 40 = 2200 kotak sel

    std::vector<int> cellCount(totalCells, 0);
    std::vector<int> cellStart(totalCells, 0);
    std::vector<int> particleCellIDs(numParticles, 0);
    std::vector<sf::Vector2f> sortedPositions(numParticles);
    std::vector<sf::Vector2f> sortedVelocities(numParticles);

    // --- [3] SETUP OPENCL BOILERPLATE ---
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernelKecepatan;
    cl::Kernel kernelPosisi;
    
    cl::Buffer bufferPos;
    cl::Buffer bufferVel;
    cl::Buffer bufferCellStart;
    cl::Buffer bufferCellCount;

    try {
        // A. Cari Platform dan Device (GPU) secara menyeluruh
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(platforms.empty()) throw std::runtime_error("Tidak ada platform OpenCL ditemukan!");
        
        cl::Device device;
        std::vector<cl::Device> devices;
        bool gpuFound = false;

        // Loop memeriksa semua platform yang terinstal di Arch Linux Anda
        for (const auto& platform : platforms) {
            // Coba cari GPU di platform ini
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            
            if (!devices.empty()) {
                device = devices[0]; // Ambil GPU pertama yang ketemu
                gpuFound = true;
                break; // Keluar loop karena GPU sudah didapatkan!
            }
        }

        // Jika di semua platform tidak ada GPU, terpaksa gunakan CPU fallback
        if (!gpuFound) {
            std::cout << "Peringatan: GPU tidak ditemukan. Menggunakan CPU fallback.\n";
            platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
            if(devices.empty()) throw std::runtime_error("Tidak ada device OpenCL apapun!");
            device = devices[0];
        }
        
        std::cout << "Menggunakan Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // B. Buat Context dan Command Queue
        context = cl::Context(device);
        cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
        queue = cl::CommandQueue(context, device, properties);

        // C. Kompilasi Kode Kernel String menjadi Program di GPU
        std::string sourceCode = readKernelSource("simulasi_fluida.cl");
        cl::Program program(context, sourceCode);
        program.build("-cl-std=CL1.2"); // Kompilasi
        kernelKecepatan = cl::Kernel(program, "hitung_kecepatan");
        kernelPosisi = cl::Kernel(program, "update_posisi");

        // D. Buat Memori Buffer
        bufferPos = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                               sizeof(sf::Vector2f) * numParticles, positions.data());
                               
        bufferVel = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                               sizeof(sf::Vector2f) * numParticles, velocities.data());
                
        bufferCellStart = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(int) * totalCells, cellStart.data());

        
        bufferCellCount = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(int) * totalCells, cellCount.data());
    } 
    catch (cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return -1; // Berhenti jika OpenCL gagal
    }

    // --- [4] SETUP JENDELA SFML ---
    sf::RenderWindow window(sf::VideoMode({windowWidth, windowHeight}), "Simulasi OpenCL GPU", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);
    sf::Clock clock;

    // --- [5] MAIN LOOP ---
    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::KeyPressed>()) {
                const auto* keyEvent = event->getIf<sf::Event::KeyPressed>();
                if (keyEvent->scancode == sf::Keyboard::Scan::Up) {
                    fluidStiffness += .01f;
                } 
                else if (keyEvent->scancode == sf::Keyboard::Scan::Down) {
                    fluidStiffness -= .01f;
                }
            }
            if (event->is<sf::Event::Closed>()) window.close();
        }

        auto startTotal = HRClock::now();
        auto startCPU = HRClock::now();

        float deltaTime = clock.restart().asSeconds();

        // ==============================================================================
        // TAHAP A: RESET GRID
        // ==============================================================================
        std::fill(cellCount.begin(), cellCount.end(), 0);
        std::fill(cellStart.begin(), cellStart.end(), 0);

        // ==============================================================================
        // TAHAP B: MAPPING PARTIKEL (Menghitung isi setiap sel kotak)
        // ==============================================================================
        mapParticlesToGrid(positions, container, cellSize, gridWidth, gridHeight, cellCount, particleCellIDs);

        // ==============================================================================
        // TAHAP C: HITUNG OFFSET & SORTING PARTIKEL (Prefix Sum)
        // ==============================================================================
        buildGridOffsetsAndSort(positions, velocities, particleCellIDs, cellCount, cellStart, sortedPositions, sortedVelocities);

        // Sekarang, karena data di CPU sudah rapi dan terurut, kita timpa array asli kita 
        // agar sinkron untuk frame berikutnya
        positions = sortedPositions;
        velocities = sortedVelocities;

        auto endCPU = HRClock::now();
        double durationCPU = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();

        // ==============================================================================
        // SEBELUM TAHAP D: UPDATE DATA BUFFER DI GPU VIA OPENCL
        // ==============================================================================
        // Kita harus meng-upload array grid baru (cellStart & cellCount) beserta posisi 
        // terurut terbaru dari CPU ke OpenCL Buffer sebelum GPU mengeksekusi rumus SPH.
        queue.enqueueWriteBuffer(bufferPos, CL_TRUE, 0, sizeof(sf::Vector2f) * numParticles, positions.data());
        queue.enqueueWriteBuffer(bufferVel, CL_TRUE, 0, sizeof(sf::Vector2f) * numParticles, velocities.data());
        queue.enqueueWriteBuffer(bufferCellStart, CL_TRUE, 0, sizeof(int) * totalCells, cellStart.data());
        queue.enqueueWriteBuffer(bufferCellCount, CL_TRUE, 0, sizeof(int) * totalCells, cellCount.data());

        // --- BACA INPUT MOUSE ---
        float mouseForce = 0.0f;
        sf::Vector2f mousePos(0.0f, 0.0f);

        // Cek apakah tombol kiri ditahan (Tarik)
        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
            mouseForce = 150.0f; // Kekuatan tarik
        } 
        // Cek apakah tombol kanan ditahan (Tolak)
        else if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right)) {
            mouseForce = -150.0f; // Kekuatan tolak (negatif)
        }

        // Jika ada tombol yang ditekan, ambil koordinat mouse-nya
        if (mouseForce != 0.0f) {
            // Dapatkan posisi mouse di layar
            sf::Vector2i pixelPos = sf::Mouse::getPosition(window);
            // Konversi ke koordinat dunia SFML yang presisi (penting untuk window dengan view)
            mousePos = window.mapPixelToCoords(pixelPos); 
        }

        float kecepatanMaksimal = 200.0f;
        
        cl::Event profileEvent; // Objek pengukur internal GPU
        double murniGPUMs;

        try {
        // Eksekusi Fisika di GPU
        updateGPU(queue, kernelKecepatan, kernelPosisi, bufferPos, bufferVel, 
                bufferCellStart, bufferCellCount, positions, velocities, container, 
                cellSize, gridWidth, gridHeight, deltaTime, fluidStiffness, 
                mousePos, mouseForce, numParticles, &profileEvent);

        queue.finish();

        profileEvent.wait();
        
        // Ambil waktu murni hardware GPU (dalam satuan nanodetik)
        cl_ulong timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong timeEnd   = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        // Konversi ke milidetik
        murniGPUMs = (timeEnd - timeStart) / 1000000.0;
        } catch (cl::Error& e) {
            std::cerr << "\n!! DETEKSI ERROR FISIKA GPU !!" << std::endl;
            std::cerr << "Fungsi OpenCL yang Gagal: " << e.what() << std::endl;
            std::cerr << "Kode Error OpenCL (ID)  : " << e.err() << std::endl;
            
            // Berikan analisis bantuan berdasarkan Kode Angka Errornya
            if (e.err() == -5)  std::cerr << "Analisis: CL_OUT_OF_RESOURCES. GPU kehabisan memori atau Kernel melakukan Out-of-Bounds array access!\n";
            if (e.err() == -36) std::cerr << "Analisis: CL_INVALID_COMMAND_QUEUE. Command Queue rusak atau macet.\n";
            if (e.err() == -58) std::cerr << "Analisis: CL_INVALID_EVENT. Objek Event Anda tidak valid/tertimpa memori lain!\n";
            
            return -1; // Matikan program dengan rapi agar tidak core dumped kaku
        }

        #pragma omp parallel for
        for (size_t i = 0; i < positions.size(); ++i) {
            // Ambil kecepatan x dan y dari partikel ke-i, lalu ubah jadi warna
            colors[i] = getSpeedColor(velocities[i].x, velocities[i].y, kecepatanMaksimal);
        }

        // --- SELESAI PENCATATAN WAKTU FRAME ---
        auto endTotal = HRClock::now();
        double durationTotal = std::chrono::duration<double, std::milli>(endTotal - startTotal).count();
        float fps = 1000.0f / durationTotal;

        // --- CETAK HASIL KE TERMINAL SETIAP SEBEKAS FRAME (Misal tiap 60 frame sekali agar tidak spamming) ---
        static int frameCounter = 0;
        if (frameCounter++ % 60 == 0) {
            std::cout << "--- BENCHMARK REPORT (OMP_THREADS = " << omp_get_max_threads() << ") ---\n"
                    << "Total Frame Time : " << durationTotal << " ms | FPS: " << fps << "\n"
                    << "CPU Grid & Sort  : " << durationCPU << " ms\n"
                    << "GPU SPH Physics  : " << murniGPUMs << " ms\n"
                    << "--------------------------------------------\n" << std::endl;
        }
        
        // Gambar di CPU/SFML
        render(window, positions, colors, container);
    }

    return 0;
}

// ==============================================================================
// FUNGSI UPDATE (DIKENDALIKAN OPENCL)
// ==============================================================================
void updateGPU (cl::CommandQueue& queue, 
                cl::Kernel& kernelKecepatan, cl::Kernel& kernelPosisi, 
                cl::Buffer& bufferPos, cl::Buffer& bufferVel,
                cl::Buffer& bufferCellStart, cl::Buffer& bufferCellCount,
                std::vector<sf::Vector2f>& positions, std::vector<sf::Vector2f>& velocities, const sf::FloatRect& container, 
                float cellSize, int gridWidth, int gridHeight, float deltaTime, float fluidStiffness,
                sf::Vector2f mousePos, float mouseForce, int numParticles, cl::Event* profEvent) 
{
    const float gravity = 9.81f * 100.0f; 
    const float radiusSPH = cellSize; // Radius pengaruh SPH disamakan dengan ukuran sel

    // Tentukan ukuran grup thread secara global untuk KEDUA kernel
    size_t local_threads = 32; // <--- UBAH ANGKA INI UNTUK BENCHMARK (32, 64, 128, 256)
    size_t global_threads = ((positions.size() + local_threads - 1) / local_threads) * local_threads;

    cl::NDRange globalSize(global_threads);
    cl::NDRange localSize(local_threads);

    // --- KERNEL 1: HITUNG KECEPATAN (LOGIKA SPH NEIGHBOR SEARCH) ---
    // Di sinilah nanti interaksi antar-partikel (densitas/tekanan fluida) dihitung
    kernelKecepatan.setArg(0, bufferPos);
    kernelKecepatan.setArg(1, bufferVel);
    kernelKecepatan.setArg(2, bufferCellStart);
    kernelKecepatan.setArg(3, bufferCellCount);
    kernelKecepatan.setArg(4, gravity);
    kernelKecepatan.setArg(5, radiusSPH);
    kernelKecepatan.setArg(6, cellSize);
    kernelKecepatan.setArg(7, gridWidth);
    kernelKecepatan.setArg(8, gridHeight);
    kernelKecepatan.setArg(9, container.position.x);  // container_left
    kernelKecepatan.setArg(10, container.position.x + container.size.x); // container_right
    kernelKecepatan.setArg(11, container.position.y); // PERBAIKAN: container_top!
    kernelKecepatan.setArg(12, container.position.y + container.size.y); // container_bottom
    kernelKecepatan.setArg(13, deltaTime); // Bergeser ke index 13
    kernelKecepatan.setArg(14, fluidStiffness);

    cl_float2 clMousePos = {mousePos.x, mousePos.y};
    kernelKecepatan.setArg(15, clMousePos);
    kernelKecepatan.setArg(16, mouseForce);
    kernelKecepatan.setArg(17, numParticles);

    // Jalankan Kernel 1 di GPU
    queue.enqueueNDRangeKernel(kernelKecepatan, cl::NullRange, globalSize, localSize, nullptr, profEvent);

    // --- KERNEL 2: UPDATE POSISI & COLLISION CONTAINER ---
    kernelPosisi.setArg(0, bufferPos);
    kernelPosisi.setArg(1, bufferVel);
    kernelPosisi.setArg(2, container.position.x); // Batas kiri fisik kotak
    kernelPosisi.setArg(3, container.position.x + container.size.x); // Batas kanan fisik kotak
    kernelPosisi.setArg(4, container.position.y + container.size.y); // Batas bawah fisik kotak
    kernelPosisi.setArg(5, deltaTime);
    kernelPosisi.setArg(6, numParticles);

    // Jalankan Kernel 2 di GPU
    queue.enqueueNDRangeKernel(kernelPosisi, cl::NullRange, globalSize, localSize, nullptr, nullptr);

    // --- TARIK DATA POSISI DAN KECEPATAN TERBARU KE RAM CPU UNTUK DIGAMBAR ---
    queue.enqueueReadBuffer(bufferPos, CL_TRUE, 0, sizeof(sf::Vector2f) * positions.size(), positions.data());
    queue.enqueueReadBuffer(bufferVel, CL_TRUE, 0, sizeof(sf::Vector2f) * velocities.size(), velocities.data());
}

// ==============================================================================
// FUNGSI RENDER (SFML)
// ==============================================================================
void render(sf::RenderWindow& window, const std::vector<sf::Vector2f>& positions, 
            const std::vector<sf::Color>& colors, const sf::FloatRect& container) 
{
    window.clear(sf::Color::Black);

    sf::RectangleShape box(container.size); 
    box.setPosition(container.position);
    box.setFillColor(sf::Color::Transparent);
    box.setOutlineColor(sf::Color::White);
    box.setOutlineThickness(2.0f);
    window.draw(box);
    
    sf::VertexArray va(sf::PrimitiveType::Triangles, positions.size() * 6);
    float size = 1.0f; 

    for (size_t i = 0; i < positions.size(); ++i) {
        sf::Vector2f pos = positions[i];
        size_t idx = i * 6; 

        sf::Vector2f topLeft(pos.x - size, pos.y - size);
        sf::Vector2f topRight(pos.x + size, pos.y - size);
        sf::Vector2f bottomLeft(pos.x - size, pos.y + size);
        sf::Vector2f bottomRight(pos.x + size, pos.y + size);

        va[idx + 0].position = topLeft;     va[idx + 1].position = topRight;    va[idx + 2].position = bottomLeft;
        va[idx + 3].position = topRight;    va[idx + 4].position = bottomRight; va[idx + 5].position = bottomLeft;
        
        for (int j = 0; j < 6; ++j) va[idx + j].color = colors[i];
    }
    window.draw(va); 
    window.display();
}

// ==============================================================================
// FUNGSI INISIALISASI
// ==============================================================================
void initializeParticles(std::vector<sf::Vector2f>& positions, std::vector<sf::Color>& colors, 
                         const sf::FloatRect& container) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distX(container.position.x, container.position.x + container.size.x);
    std::uniform_real_distribution<float> distY(container.position.y, container.position.y + container.size.y);

    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i].x = distX(gen);
        positions[i].y = distY(gen);
        colors[i] = sf::Color::Green;
    }
}

// Fungsi untuk Tahap B: Mapping Partikel ke Sel Grid
void mapParticlesToGrid(const std::vector<sf::Vector2f>& positions, 
                        const sf::FloatRect& container, 
                        float cellSize, 
                        int gridWidth, 
                        int gridHeight,
                        std::vector<int>& cellCount, 
                        std::vector<int>& particleCellIDs) 
{
    // Pastikan ukuran vector penampung ID sel per partikel sama dengan jumlah partikel
    particleCellIDs.resize(positions.size());

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < positions.size(); ++i) {
        sf::Vector2f pos = positions[i];

        // 1. Hitung posisi relatif partikel di dalam kontainer
        float relativeX = pos.x - container.position.x;
        float relativeY = pos.y - container.position.y;

        // 2. Cari indeks kolom (x) dan baris (y) grid
        int cellX = static_cast<int>(relativeX / cellSize);
        int cellY = static_cast<int>(relativeY / cellSize);

        // 3. Batasi (Clamp) indeks agar tidak keluar dari batas array akibat eror floating-point fisik
        if (cellX < 0) cellX = 0;
        if (cellX >= gridWidth)  cellX = gridWidth - 1;
        if (cellY < 0) cellY = 0;
        if (cellY >= gridHeight) cellY = gridHeight - 1;

        // 4. Hitung ID Sel linear 1D
        int cellID = cellY * gridWidth + cellX;

        // 5. Catat hasilnya
        particleCellIDs[i] = cellID; // Partikel ke-i berada di sel 'cellID'

        #pragma omp atomic
        cellCount[cellID]++;         // Jumlah partikel di sel 'cellID' bertambah 1
    }
}

void buildGridOffsetsAndSort(const std::vector<sf::Vector2f>& positions,
                             const std::vector<sf::Vector2f>& velocities,
                             const std::vector<int>& particleCellIDs,
                             const std::vector<int>& cellCount,
                             std::vector<int>& cellStart,
                             std::vector<sf::Vector2f>& sortedPositions,
                             std::vector<sf::Vector2f>& sortedVelocities) 
{
    int totalCells = cellCount.size();
    int numParticles = positions.size();

    // 1. HITUNG PREFIX SUM (Daftar Isi)
    int accumulator = 0;
    for (int i = 0; i < totalCells; ++i) {
        cellStart[i] = accumulator;
        accumulator += cellCount[i]; // Geser indeks awal untuk sel berikutnya
    }

    // 2. SORTING PARTIKEL BERDASARKAN SEL
    // Kita buat array salinan sementara dari cellStart khusus untuk tracking penempatan indeks
    std::vector<int> currentCellOffset = cellStart;

    sortedPositions.resize(numParticles);
    sortedVelocities.resize(numParticles);

    for (int i = 0; i < numParticles; ++i) {
        int cellID = particleCellIDs[i];      // Cari tahu partikel i ada di sel mana
        int targetIndex = currentCellOffset[cellID]; // Cari indeks baris memori globalnya

        // Pindahkan data partikel ke array baru yang terurut rapi
        sortedPositions[targetIndex] = positions[i];
        sortedVelocities[targetIndex] = velocities[i];

        // Increment offset sel tersebut agar partikel berikutnya di sel yang sama ditaruh di sebelahnya
        currentCellOffset[cellID]++;
    }
}

sf::Color getSpeedColor(float vx, float vy, float maxSpeed) {
    // 1. Hitung kelajuan partikel menggunakan rumus Pythagoras
    float speed = std::sqrt(vx * vx + vy * vy);

    // 2. Normalisasi kecepatan ke rentang [0.0 sampai 1.0]
    float t = speed / maxSpeed;
    
    // Batasi nilai t agar tidak kurang dari 0 dan tidak melebihi 1 (Clamping)
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    int r = 0, g = 0, b = 0; // Komponen warna biru (B) selalu 0

    // 3. Logika Interpolasi Linier 3 Fase (Mulus)
    if (t < 0.333f) {
        // FASE 1: Sangat Lambat ke Lambat (Biru ke Hijau)
        // Skalakan t agar berjalan dari 0.0 sampai 1.0 di dalam segmen ini
        float factor = t / 0.333f; 
        
        r = 0;
        g = static_cast<int>(factor * 255.0f);        // Hijau naik
        b = static_cast<int>((1.0f - factor) * 255.0f); // Biru turun
    } 
    else if (t < 0.666f) {
        // FASE 2: Lambat ke Sedang (Hijau ke Kuning)
        float factor = (t - 0.333f) / 0.333f;
        
        r = static_cast<int>(factor * 255.0f);        // Merah naik (membentuk kuning)
        g = 255;                                      // Hijau menetap di puncak
        b = 0;
    } 
    else {
        // FASE 3: Sedang ke Cepat (Kuning ke Merah)
        float factor = (t - 0.666f) / 0.334f;
        
        r = 255;                                      // Merah menetap di puncak
        g = static_cast<int>((1.0f - factor) * 255.0f); // Hijau turun (menyisakan merah murni)
        b = 0;
    }

    return sf::Color(r, g, b);
}