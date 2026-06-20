/**
 * ==============================================================================
 * PROYEK TUGAS BESAR: SIMULASI FLUIDA HYBRID (CPU-GPU) BERBASIS SPH & NAVIER-STOKES
 * ==============================================================================
 * Deskripsi:
 * Program ini mensimulasikan dinamika fluida tidak-termampatkan menggunakan metode 
 * Smoothed Particle Hydrodynamics (SPH) yang diturunkan dari Persamaan Navier-Stokes.
 * * Arsitektur Komputasi Paralel (Hybrid):
 * 1. CPU (OpenMP): Bertanggung jawab atas penataan grid spasial (Spatial Hashing), 
 * perhitungan peta sel partikel, prefix sum, dan pewarnaan partikel (Color Interpolation).
 * 2. GPU (OpenCL): Bertanggung jawab penuh atas pemrosesan akselerasi fisik tingkat tinggi
 * (Looping interaksi tetangga SPH, Gradien Tekanan, Viskositas, dan Integrasi Posisi).
 * * Library Grafis: SFML (Simple and Fast Multimedia Library) - Primitive Triangles VA.
 * ==============================================================================
 */

#define CL_HPP_TARGET_OPENCL_VERSION 200 // Menargetkan spesifikasi standar runtime OpenCL 2.0
#define CL_HPP_ENABLE_EXCEPTIONS        // Mengaktifkan mekanisme try-catch untuk penanganan eror OpenCL
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
#include <omp.h>                         // Header OpenMP untuk komputasi paralel multi-core CPU
#include <chrono>                        // Untuk pengukuran instrumen waktu profiler presisi tinggi

// Fungsi pembantu (helper) untuk membaca file kode sumber kernel OpenCL (.cl)
std::string readKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// Alias untuk kemudahan pembacaan sistem clock resolusi tinggi
using HRClock = std::chrono::high_resolution_clock;

// ==============================================================================
// DEKLARASI PROTOTIPE FUNGSI UTAMA PIPELINE SIMULASI
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

// Dimensi resolusi jendela utama aplikasi
unsigned int windowWidth = 1800;
unsigned int windowHeight = 1100;

// ==============================================================================
// FUNGSI UTAMA PROGRAM (MAIN ENTRY POINT)
// ==============================================================================
int main() {
    // --- [1] INISIALISASI PEMBATAS WADAH (WADAH SIMULASI) ---
    // Menyisakan margin 50 piksel di setiap sisi jendela sebagai ruang kosong visual
    sf::Vector2f containerSize(((float) windowWidth) - 100, ((float) windowHeight) - 100); 
    sf::Vector2f containerPos((windowWidth - containerSize.x) / 2.0f, (windowHeight - containerSize.y) / 2.0f);
    sf::FloatRect container(containerPos, containerSize);

    // --- [2] ALOKASI MEMORI STRUKTUR PARTIKEL & GRID DATA (Sisi Host CPU) ---
    int numParticles = 50000;      // Skala masif: Memanfaatkan ribuan core GPU secara optimal
    float fluidStiffness = .01f;   // Konstanta k awal (Gas Ideal Equation of State)
    
    // Implementasi model SoA (Structure of Arrays) untuk menjamin coalesced memory access di VRAM GPU
    std::vector<sf::Vector2f> positions(numParticles);
    std::vector<sf::Color> colors(numParticles);
    std::vector<sf::Vector2f> velocities(numParticles, {0.0f, 0.0f});
    
    // Tebar partikel secara acak di dalam kontainer pada awal program
    initializeParticles(positions, colors, container);

    // Konfigurasi Parameter Grid Spasial untuk Optimasi Pencarian Tetangga O(n)
    const float cellSize = 100.0f; // Ukuran kotak grid disamakan dengan radius interaksi (h) SPH
    int gridWidth  = static_cast<int>(container.size.x / cellSize); 
    int gridHeight = static_cast<int>(container.size.y / cellSize); 
    int totalCells = gridWidth * gridHeight; 

    // Buffer CPU pendukung algoritma Counting Sort dan Spatial Hashing
    std::vector<int> cellCount(totalCells, 0);                 // Menyimpan jumlah partikel per sel
    std::vector<int> cellStart(totalCells, 0);                 // Menyimpan indeks awal partikel per sel di memori global
    std::vector<int> particleCellIDs(numParticles, 0);         // Pemetaan 1-to-1 partikel ke ID sel grid
    std::vector<sf::Vector2f> sortedPositions(numParticles);   // Buffer penampung posisi terurut koheren
    std::vector<sf::Vector2f> sortedVelocities(numParticles);  // Buffer penampung kecepatan terurut koheren

    // --- [3] ALUR INISIALISASI OPENCL (BOILERPLATE PLATFORM & DEVICE) ---
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernelKecepatan;
    cl::Kernel kernelPosisi;
    
    cl::Buffer bufferPos;
    cl::Buffer bufferVel;
    cl::Buffer bufferCellStart;
    cl::Buffer bufferCellCount;

    try {
        // A. Enumerasi Platform OpenCL (Mendeteksi driver AMD, NVIDIA, Intel, atau Mesa)
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(platforms.empty()) throw std::runtime_error("Tidak ada platform OpenCL ditemukan!");
        
        cl::Device device;
        std::vector<cl::Device> devices;
        bool gpuFound = false;

        // Iterasi penelusuran platform untuk memprioritaskan Kartu Grafis Diskrit (GPU)
        for (const auto& platform : platforms) {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            if (!devices.empty()) {
                device = devices[0]; // Pilih GPU pertama yang tervalidasi aktif
                gpuFound = true;
                break; 
            }
        }

        // Mekanisme Fallback: Gunakan CPU jika perangkat tidak memiliki GPU diskrit/terintegrasi
        if (!gpuFound) {
            std::cout << "Peringatan: GPU tidak ditemukan. Menggunakan CPU fallback.\n";
            platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
            if(devices.empty()) throw std::runtime_error("Tidak ada device OpenCL apapun!");
            device = devices[0];
        }
        
        std::cout << "Menggunakan Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // B. Konstruksi Context dan Command Queue dengan Fitur Profiling Hardware Aktif
        context = cl::Context(device);
        cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE; // Diperlukan untuk benchmarking murni GPU
        queue = cl::CommandQueue(context, device, properties);

        // C. Pembacaan, Kompilasi Runtime JIT (Just-In-Time), dan Registrasi Objek Kernel
        std::string sourceCode = readKernelSource("simulasi_fluida.cl");
        cl::Program program(context, sourceCode);
        program.build("-cl-std=CL1.2"); // Kompilasi menggunakan standar OpenCL 1.2 yang universal
        
        kernelKecepatan = cl::Kernel(program, "hitung_kecepatan");
        kernelPosisi = cl::Kernel(program, "update_posisi");

        // D. Alokasi Memori Global (Device Buffer VRAM) dan Transfer Data Awal dari RAM Host
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
        std::cerr << "OpenCL Error Fatal: " << e.what() << " (" << e.err() << ")" << std::endl;
        return -1; 
    }

    // --- [4] KONFIGURASI WINDOWS TAMPILAN JENDELA SFML ---
    sf::RenderWindow window(sf::VideoMode({windowWidth, windowHeight}), "Simulasi Fluida Hybrid GPU OpenCL", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60); // Mengunci simulasi pada batas atas layar monitor standar (60 FPS)
    sf::Clock clock;              // Mengukur rentang waktu riil per frame (dt)

    // --- [5] SIKLUS LOOP UTAMA SIMULASI (GAME LOOP) ---
    while (window.isOpen()) {
        // Event Handler: Deteksi tombol keyboard dan penutupan aplikasi
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::KeyPressed>()) {
                const auto* keyEvent = event->getIf<sf::Event::KeyPressed>();
                // Interaktivitas: Mengubah kekakuan fluida (fluidStiffness) secara real-time via keyboard
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

        // Cari delta time riil dari frame sebelumnya (dt komputasi fisika)
        float deltaTime = clock.restart().asSeconds();

        // ==============================================================================
        // TAHAP A: CLEANUP GRID (CPU MULTI-THREAD)
        // Reset akumulator data grid frame sebelumnya menjadi nol mutlak
        // ==============================================================================
        std::fill(cellCount.begin(), cellCount.end(), 0);
        std::fill(cellStart.begin(), cellStart.end(), 0);

        // ==============================================================================
        // TAHAP B: MAPPING SPASIAL PARTIKEL TO GRID SEL (CPU OpenMP Paralel)
        // Memetakan posisi kontinu 2D partikel ke koordinat diskrit kotak sel grid 1D
        // ==============================================================================
        mapParticlesToGrid(positions, container, cellSize, gridWidth, gridHeight, cellCount, particleCellIDs);

        // ==============================================================================
        // TAHAP C: ALGORITMA PREFIX SUM & MEMORY SORTING KOHEREN (Sisi CPU Host)
        // Mengurutkan posisi memori partikel agar data yang berada di sel grid yang sama 
        // terletak berdampingan di memori RAM. Hal ini sangat krusial untuk memicu 
        // mekanisme "L1/L2 Cache Hitting" saat GPU OpenCL melakukan pembacaan tetangga.
        // ==============================================================================
        buildGridOffsetsAndSort(positions, velocities, particleCellIDs, cellCount, cellStart, sortedPositions, sortedVelocities);

        // Sinkronisasi data memori utama internal host CPU
        positions = sortedPositions;
        velocities = sortedVelocities;

        auto endCPU = HRClock::now();
        double durationCPU = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();

        // ==============================================================================
        // TRANSFER MEMORI: RE-UPLOAD HASIL URUTAN CPU KE BUFFER MEMORI EMAS GPU
        // ==============================================================================
        queue.enqueueWriteBuffer(bufferPos, CL_TRUE, 0, sizeof(sf::Vector2f) * numParticles, positions.data());
        queue.enqueueWriteBuffer(bufferVel, CL_TRUE, 0, sizeof(sf::Vector2f) * numParticles, velocities.data());
        queue.enqueueWriteBuffer(bufferCellStart, CL_TRUE, 0, sizeof(int) * totalCells, cellStart.data());
        queue.enqueueWriteBuffer(bufferCellCount, CL_TRUE, 0, sizeof(int) * totalCells, cellCount.data());

        // --- MANIPULASI GAYA INTERAKTIF: PENULISAN VEKTOR MOUSE ---
        float mouseForce = 0.0f;
        sf::Vector2f mousePos(0.0f, 0.0f);

        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
            mouseForce = 150.0f;  // Gaya Hisap (Gravitasi Lubang Hitam Buatan)
        } 
        else if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right)) {
            mouseForce = -150.0f; // Gaya Tolak (Ledakan Tekanan Radial)
        }

        if (mouseForce != 0.0f) {
            sf::Vector2i pixelPos = sf::Mouse::getPosition(window);
            mousePos = window.mapPixelToCoords(pixelPos); // Pemetaan presisi koordinat layar ke koordinat simulasi
        }

        float kecepatanMaksimal = 200.0f; // Batas acuan interpolasi gradasi warna kelajuan
        cl::Event profileEvent;           // Menampung payload data pencatatan waktu internal hardware GPU
        double murniGPUMs;

        try {
            // ==============================================================================
            // TAHAP D: EKSEKUSI PIPELINE FISIKA SPH DI HARDWARE UTAMA GPU via OpenCL
            // Memanggil fungsi pengeksekusi kernel ganda OpenCL
            // ==============================================================================
            updateGPU(queue, kernelKecepatan, kernelPosisi, bufferPos, bufferVel, 
                      bufferCellStart, bufferCellCount, positions, velocities, container, 
                      cellSize, gridWidth, gridHeight, deltaTime, fluidStiffness, 
                      mousePos, mouseForce, numParticles, &profileEvent);

            queue.finish(); // Memaksa CPU menunggu hingga GPU menyelesaikan seluruh antrean instruksi
            profileEvent.wait();
            
            // Mengambil data murni stempel waktu register hardware internal GPU (dalam satuan nanodetik)
            cl_ulong timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong timeEnd   = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            murniGPUMs = (timeEnd - timeStart) / 1000000.0; // Konversi skalar ke milidetik
        } 
        catch (cl::Error& e) {
            std::cerr << "\n!! DETEKSI ERROR EKSEKUSI RUNTIME HARDWARE GPU !!" << std::endl;
            std::cerr << "Fungsi OpenCL Gagal: " << e.what() << " | ID Error: " << e.err() << std::endl;
            if (e.err() == -5)  std::cerr << "Analisis Diagnostik: CL_OUT_OF_RESOURCES. Terjadi Out-of-Bounds Memory Access pada pointer array di Kernel!\n";
            return -1; 
        }

        // ==============================================================================
        // TAHAP E: VISUALISASI - INTERPOLASI KELAJUAN PARTIKEL (CPU OpenMP parallel for)
        // Membagi porsi penghitungan warna partikel berdasarkan resultan kecepatan secara
        // paralel di seluruh core prosesor CPU.
        // ==============================================================================
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < positions.size(); ++i) {
            colors[i] = getSpeedColor(velocities[i].x, velocities[i].y, kecepatanMaksimal);
        }

        // --- KALKULASI & PENYAJIAN INSTRUMEN BENCHMARK HPC ---
        auto endTotal = HRClock::now();
        double durationTotal = std::chrono::duration<double, std::milli>(endTotal - startTotal).count();
        float fps = 1000.0f / durationTotal;

        // Reduksi keluaran terminal: Hanya mencetak performa komputasi setiap 60 frame sekali
        static int frameCounter = 0;
        if (frameCounter++ % 60 == 0) {
            std::cout << "--- BENCHMARK REPORT (OMP_THREADS = " << omp_get_max_threads() << ") ---\n"
                    << "Total Frame Time : " << durationTotal << " ms | FPS: " << fps << "\n"
                    << "CPU Grid & Sort  : " << durationCPU << " ms\n"
                    << "GPU SPH Physics  : " << murniGPUMs << " ms\n"
                    << "--------------------------------------------\n" << std::endl;
        }
        
        // Panggil fungsi render grafis SFML
        render(window, positions, colors, container);
    }

    return 0;
}

// ==============================================================================
// IMPLEMENTASI FUNGSI MANAJEMEN EKSEKUSI THREAD GPU (OPENCL WRAPPER)
// ==============================================================================
void updateGPU (cl::CommandQueue& queue, 
                cl::Kernel& kernelKecepatan, cl::Kernel& kernelPosisi, 
                cl::Buffer& bufferPos, cl::Buffer& bufferVel,
                cl::Buffer& bufferCellStart, cl::Buffer& bufferCellCount,
                std::vector<sf::Vector2f>& positions, std::vector<sf::Vector2f>& velocities, const sf::FloatRect& container, 
                float cellSize, int gridWidth, int gridHeight, float deltaTime, float fluidStiffness,
                sf::Vector2f mousePos, float mouseForce, int numParticles, cl::Event* profEvent) 
{
    const float gravity = 9.81f * 100.0f; // Skala akselerasi gravitasi disesuaikan dengan konversi rasio piksel
    const float radiusSPH = cellSize;     // Radius dukung kernel (h) SPH murni disinkronkan dengan ukuran kotak grid

    // --- STRUKTUR UKURAN GROUP THREAD GPU (HPC BENCHMARK CONFIGURATION) ---
    size_t local_threads = 32;            // Ukuran fiksasi satu blok Wavefront/Warp hardware (32 thread per Compute Unit)
    // Melakukan padding pembulatan ke atas kelipatan 32 agar total thread mencakup seluruh jumlah partikel tanpa sisa
    size_t global_threads = ((positions.size() + local_threads - 1) / local_threads) * local_threads;

    cl::NDRange globalSize(global_threads);
    cl::NDRange localSize(local_threads);

    // --------------------------------================================--------------
    // KONFIGURASI PARAMETER & ARGUMEN KERNEL 1: NAVIER-STOKES SPEED SOLVER
    // --------------------------------================================--------------
    kernelKecepatan.setArg(0, bufferPos);
    kernelKecepatan.setArg(1, bufferVel);
    kernelKecepatan.setArg(2, bufferCellStart);
    kernelKecepatan.setArg(3, bufferCellCount);
    kernelKecepatan.setArg(4, gravity);
    kernelKecepatan.setArg(5, radiusSPH);
    kernelKecepatan.setArg(6, cellSize);
    kernelKecepatan.setArg(7, gridWidth);
    kernelKecepatan.setArg(8, gridHeight);
    kernelKecepatan.setArg(9, container.position.x);                     // Batas kiri wadah
    kernelKecepatan.setArg(10, container.position.x + container.size.x); // Batas kanan wadah
    kernelKecepatan.setArg(11, container.position.y);                     // Batas atas wadah (langit-langit)
    kernelKecepatan.setArg(12, container.position.y + container.size.y); // Batas bawah wadah (lantai fisik)
    kernelKecepatan.setArg(13, deltaTime);
    kernelKecepatan.setArg(14, fluidStiffness);

    cl_float2 clMousePos = {mousePos.x, mousePos.y};                      // Konversi struct tipe data khusus OpenCL
    kernelKecepatan.setArg(15, clMousePos);
    kernelKecepatan.setArg(16, mouseForce);
    kernelKecepatan.setArg(17, numParticles);

    // Kirim instruksi eksekusi Kernel 1 ke dalam antrean GPU (Pencatatan waktu diwakili oleh profEvent)
    queue.enqueueNDRangeKernel(kernelKecepatan, cl::NullRange, globalSize, localSize, nullptr, profEvent);

    // --------------------------------================================--------------
    // KONFIGURASI PARAMETER & ARGUMEN KERNEL 2: ADVEKSI POSISI & HANDLE BOUNDARY DETECTOR
    // --------------------------------================================--------------
    kernelPosisi.setArg(0, bufferPos);
    kernelPosisi.setArg(1, bufferVel);
    kernelPosisi.setArg(2, container.position.x); 
    kernelPosisi.setArg(3, container.position.x + container.size.x); 
    kernelPosisi.setArg(4, container.position.y + container.size.y); 
    kernelPosisi.setArg(5, deltaTime);
    kernelPosisi.setArg(6, numParticles);

    // Kirim instruksi eksekusi Kernel 2 langsung ke antrean GPU
    queue.enqueueNDRangeKernel(kernelPosisi, cl::NullRange, globalSize, localSize, nullptr, nullptr);

    // --------------------------------================================--------------
    // ASINKRONUS READ-BACK BUFFER: DOWNLOAD HASIL FISIKA GPU KEMBALI KE RAM CPU INTERFACES
    // --------------------------------================================--------------
    queue.enqueueReadBuffer(bufferPos, CL_TRUE, 0, sizeof(sf::Vector2f) * positions.size(), positions.data());
    queue.enqueueReadBuffer(bufferVel, CL_TRUE, 0, sizeof(sf::Vector2f) * velocities.size(), velocities.data());
}

// ==============================================================================
// FUNGSI RENDER GRAFIS - OPTIMASI VERTEX ARRAY SFML (BATCHED RENDERING)
// Menggambar puluhan ribu objek partikel sekaligus dalam 1 kali instruksi draw 
// (Draw Call) menggunakan struktur data Quad berbasis primitive Triangles.
// ==============================================================================
void render(sf::RenderWindow& window, const std::vector<sf::Vector2f>& positions, 
            const std::vector<sf::Color>& colors, const sf::FloatRect& container) 
{
    window.clear(sf::Color::Black); // Menghapus frame lama dengan latar hitam pekat

    // Gambar garis outline kaku wadah batas kontainer
    sf::RectangleShape box(container.size); 
    box.setPosition(container.position);
    box.setFillColor(sf::Color::Transparent);
    box.setOutlineColor(sf::Color::White);
    box.setOutlineThickness(2.0f);
    window.draw(box);
    
    // Alokasi batched vertex memory (1 partikel dirender sebagai kotak 2-Triangle = 6 simpul titik)
    sf::VertexArray va(sf::PrimitiveType::Triangles, positions.size() * 6);
    float size = 1.0f; // Setengah ukuran lebar visual partikel butiran air (dalam piksel)

    for (size_t i = 0; i < positions.size(); ++i) {
        sf::Vector2f pos = positions[i];
        size_t idx = i * 6; // Hitung offset penulisan memori simpul partikel ke-i

        // Rekonstruksi geometri bentuk persegi mengitari titik pusat koordinat partikel
        sf::Vector2f topLeft(pos.x - size, pos.y - size);
        sf::Vector2f topRight(pos.x + size, pos.y - size);
        sf::Vector2f bottomLeft(pos.x - size, pos.y + size);
        sf::Vector2f bottomRight(pos.x + size, pos.y + size);

        // Segitiga 1
        va[idx + 0].position = topLeft;     va[idx + 1].position = topRight;    va[idx + 2].position = bottomLeft;
        // Segitiga 2
        va[idx + 3].position = topRight;    va[idx + 4].position = bottomRight; va[idx + 5].position = bottomLeft;
        
        // Suntikkan data pewarnaan interpolasi kelajuan ke seluruh 6 simpul kuadrat tersebut
        for (int j = 0; j < 6; ++j) va[idx + j].color = colors[i];
    }
    
    window.draw(va); // Eksekusi 1 single batched draw call ke driver kartu grafis
    window.display();
}

// ==============================================================================
// FUNGSI INIT SEED TABURAN RANDOM PARTIKEL
// ==============================================================================
void initializeParticles(std::vector<sf::Vector2f>& positions, std::vector<sf::Color>& colors, 
                         const sf::FloatRect& container) 
{
    std::random_device rd;
    std::mt19937 gen(rd()); // Generator bilangan acak berbasis algoritma Mersenne Twister
    // Batasi distribusi rentang koordinat ketat murni di dalam area interior wadah kontainer
    std::uniform_real_distribution<float> distX(container.position.x + 10.0f, container.position.x + container.size.x - 10.0f);
    std::uniform_real_distribution<float> distY(container.position.y + 10.0f, container.position.y + container.size.y - 10.0f);

    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i].x = distX(gen);
        positions[i].y = distY(gen);
        colors[i] = sf::Color::Green; // Default awal warna hijau
    }
}

// ==============================================================================
// TAHAP B (DETAIL): SPATIAL HASHING MAPPING PIPELINE (CPU OpenMP Parallel)
// Fungsi membelah matriks ruang kontinu ke sel grid linear untuk melacak kepadatan sel.
// ==============================================================================
void mapParticlesToGrid(const std::vector<sf::Vector2f>& positions, 
                        const sf::FloatRect& container, 
                        float cellSize, 
                        int gridWidth, 
                        int gridHeight,
                        std::vector<int>& cellCount, 
                        std::vector<int>& particleCellIDs) 
{
    particleCellIDs.resize(positions.size());

    // Membuka loop paralel untuk dikerjakan bersamaan di seluruh thread inti CPU Anda
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < positions.size(); ++i) {
        sf::Vector2f pos = positions[i];

        // 1. Transformasi translasi posisi koordinat absolut dunia ke koordinat relatif lokal wadah
        float relativeX = pos.x - container.position.x;
        float relativeY = pos.y - container.position.y;

        // 2. Diskritasi pembagian ruang skalar koordinat menjadi penunjuk indeks 2D kolom/baris sel
        int cellX = static_cast<int>(relativeX / cellSize);
        int cellY = static_cast<int>(relativeY / cellSize);

        // 3. Clamping pengaman untuk mencegah kegagalan fatal memory segfault akibat letak presisi floating point
        if (cellX < 0) cellX = 0;
        if (cellX >= gridWidth)  cellX = gridWidth - 1;
        if (cellY < 0) cellY = 0;
        if (cellY >= gridHeight) cellY = gridHeight - 1;

        // 4. Konversi proyeksi matriks sel grid 2D $(x,y)$ ke representasi alamat Array Linear 1D
        int cellID = cellY * gridWidth + cellX;

        particleCellIDs[i] = cellID; // Pasangkan relasi: Partikel ke-i menempati kotak alamat 'cellID'

        // Menggunakan direktif atomic untuk menjamin inkremen data aman dari bentrokan thread paralel (Race Condition)
        #pragma omp atomic
        cellCount[cellID]++;         
    }
}

// ==============================================================================
// TAHAP C (DETAIL): RADIX-BASED DISCRETE COHERENT SORTING PIPELINE
// Menyusun ulang array posisi dan kecepatan berdasarkan urutan linear sel grid.
// Sesuai dasar teori High Performance Computing (HPC), data yang berdekatan secara 
// spasial wajib ditata berdekatan di sektor baris memori (Memory Coherency).
// ==============================================================================
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

    // 1. HITUNG ALGORITMA PREFIX SUM (Eksklusif Scan untuk membangun tabel alamat indeks)
    int accumulator = 0;
    for (int i = 0; i < totalCells; ++i) {
        cellStart[i] = accumulator;
        accumulator += cellCount[i]; // Akumulasikan lompatan baris indeks sel secara linear berkelanjutan
    }

    // 2. RE-ORDER MEMORI (Counting Sort Tahap Akhir)
    std::vector<int> currentCellOffset = cellStart; // Duplikasi array penunjuk indeks awal sel

    sortedPositions.resize(numParticles);
    sortedVelocities.resize(numParticles);

    // Menyusun ulang lokasi elemen array ke letak baris baru yang koheren 
    for (int i = 0; i < numParticles; ++i) {
        int cellID = particleCellIDs[i];             // Deteksi lokasi sel partikel i
        int targetIndex = currentCellOffset[cellID]; // Ambil ketersediaan ruang alamat di baris baru

        // Pindahkan properti partikel ke array baru secara urut berdampingan
        sortedPositions[targetIndex] = positions[i];
        sortedVelocities[targetIndex] = velocities[i];

        // Majukan offset penunjuk sel sejauh +1 elemen untuk mengalokasikan ruang partikel berikutnya
        currentCellOffset[cellID]++;
    }
}

// ==============================================================================
// FUNGSI INTERPOLASI WARNA (3-PHASE LINEAR COLOR GRADIENT SOLVER)
// Menghasilkan warna dinamik partikel berdasarkan besaran resultan kecepatan.
// Spektrum Gradasi: Biru (Tenang) -> Hijau (Mengalir) -> Kuning -> Merah (Sangat Cepat)
// ==============================================================================
sf::Color getSpeedColor(float vx, float vy, float maxSpeed) {
    float speed = std::sqrt(vx * vx + vy * vy); // Magnitudo vektor kecepatan via rumus Pythagoras
    float t = speed / maxSpeed;                 // Normalisasi nilai t ke skala skalar antara [0.0f - 1.0f]
    
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    int r = 0, g = 0, b = 0;

    if (t < 0.333f) {
        // SEGMENTASI FASE 1: Kelajuan Rendah (Transisi warna Biru menuju Hijau)
        float factor = t / 0.333f; 
        r = 0;
        g = static_cast<int>(factor * 255.0f);        
        b = static_cast<int>((1.0f - factor) * 255.0f); 
    } 
    else if (t < 0.666f) {
        // SEGMENTASI FASE 2: Kelajuan Menengah (Transisi warna Hijau menuju Kuning)
        float factor = (t - 0.333f) / 0.333f;
        r = static_cast<int>(factor * 255.0f);        
        g = 255;                                      
        b = 0;
    } 
    else {
        // SEGMENTASI FASE 3: Kelajuan Tinggi / Turbulen (Transisi warna Kuning menuju Merah murni)
        float factor = (t - 0.666f) / 0.334f;
        r = 255;                                      
        g = static_cast<int>((1.0f - factor) * 255.0f); 
        b = 0;
    }

    return sf::Color(r, g, b);
}