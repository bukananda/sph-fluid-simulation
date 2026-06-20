/**
 * ==============================================================================
 * KODE SUMBER KERNEL GPU: RUNTIME FISIKA FLUIDA SPH (OPENCL HARDWARE ACCELERATION)
 * ==============================================================================
 * Deskripsi:
 * File ini berisi fungsi-fungsi komputasi paralel masif yang dieksekusi langsung
 * di core hardware GPU. Mengimplementasikan metode Smoothed Particle Hydrodynamics 
 * (SPH) dengan arsitektur pencarian tetangga berbasis Spatial Hashing Grid.
 * ==============================================================================
 */

// ==============================================================================
// FUNGSI PEMBANTU: MATRIKS REKONSTRUKSI FUNGSI KERNEL SPH (SMOOTHING FUNCTIONS)
// ==============================================================================
/**
 * Diberikan jarak skalar antarpartikel, fungsi ini menghitung magnitudo gaya tekan
 * buatan berdasarkan variasi penurunan bentuk kurva matematis.
 * @param dist Jarak murni hasil akar Pythagoras antarpartikel (r)
 * @param h Radius pengaruh jangkauan smoothing kernel (smoothing length)
 * @param stiffness Konstanta faktor kekakuan tolak-menolak cairan (k)
 */
float calculate_smoothing_force(float dist, float h, float stiffness) {
    if (dist < h && dist > 0.0001f) {
        float diff = h - dist;
        
        // --- TUNING BENTUK KURVA MATEMATIS SPH ---
        // Pilihan 1: Kurva Linear (Penurunan gradien konstan - keras dan tajam)
        // return diff * stiffness; 
        
        // Pilihan 2: Kurva Kuadratik "Spiky" (Rekomendasi Grafika Komputer)
        // Karakteristik: Gaya tolak meningkat drastis secara kuadratik saat partikel 
        // saling mendekat, sangat efektif mencegah penggumpalan partikel yang tumpang tindih.
        return pow(diff, 2) * stiffness; 

        // Pilihan 3: Kurva Kubik (Transisi kelandaian sangat halus, ideal untuk gas/asap)
        // return (diff * diff * diff) * stiffness;
    }
    return 0.0f;
}

// ==============================================================================
// KERNEL HARDWARE 1: INTEGRASI GAYA EKSTERNAL & SOLVER TETANGGA SPASIAL O(n)
// ==============================================================================
/**
 * Thread GPU mengeksekusi perhitungan interaksi gaya hidrostatik secara paralel.
 * Memanfaatkan struktur data tabel hash spasial untuk memotong kompleksitas dari O(n^2) menjadi O(n).
 */
__kernel void hitung_kecepatan(__global float2* positions,
                               __global float2* velocities,
                               __global const int* cellStart,
                               __global const int* cellCount,
                               const float gravity,
                               const float h, 
                               const float cellSize,
                               const int gridWidth,
                               const int gridHeight,
                               const float container_left,
                               const float container_right,
                               const float container_top,
                               const float container_bottom,
                               const float deltaTime,
                               const float stiffness,
                               const float2 mousePos,
                               const float mouseForce,
                               const int numParticles) 
{
    // Dapatkan indeks linear global yang unik untuk thread GPU saat ini
    int id = get_global_id(0);
    
    // PADDING PROTECTION: Mencegah thread hantu (thread pembulatan ke atas) mengakses memori di luar array asli
    if (id >= numParticles) return; 

    // Pemuatan data koordinat partikel target dari Global Memory ke privat register thread (Akses Instan)
    float2 my_pos = positions[id];
    
    // --- UNSUR NAVIER-STOKES: GAYA EKSTERNAL ---
    // Mengaplikasikan percepatan gravitasi makroskopis konstan ke sumbu vertikal (Y)
    velocities[id].y += gravity * deltaTime;
    
    // Damping Faktor: Simulasi gesekan/viskositas lingkungan global artifisial untuk menjaga kestabilan energi kinetik
    velocities[id] *= 0.99f; 

    // ==============================================================================
    // FITUR INTERAKSI MOUSE REKURSAL RADIAL
    // ==============================================================================
    if (mouseForce != 0.0f) {
        float m_dx = my_pos.x - mousePos.x;
        float m_dy = my_pos.y - mousePos.y;
        float m_dist_sq = (m_dx * m_dx) + (m_dy * m_dy);
        
        float mouseRadius = 100.0f; // Batas wilayah jangkauan gaya interaktif kursor mouse

        // Proteksi batas dalam (m_dist_sq > 1.0f) untuk menghindari pembagian dengan angka nol (Division by Zero)
        if (m_dist_sq < mouseRadius * mouseRadius && m_dist_sq > 1.0f) {
            float m_dist = sqrt(m_dist_sq);
            float eps = 0.001f; // Angka epsilon pengaman kestabilan nilai desimal
            
            // Profiling Gaya: Kekuatan berbanding terbalik dengan jarak (semakin dekat, semakin ekstrem)
            float forceMagnitude = (mouseRadius - m_dist) * mouseForce * deltaTime;
            
            // Normalisasi normalisasi vektor arah gaya (Direction Vector)
            float dirX = m_dx / (m_dist + eps);
            float dirY = m_dy / (m_dist + eps);
            
            // Penerapan Aksi: Mengurangi/menambah komponen kecepatan vektor partikel
            velocities[id].x -= dirX * forceMagnitude;
            velocities[id].y -= dirY * forceMagnitude;
        }
    }

    // ==============================================================================
    // 1. SPH BOUNDARY COLLISION (Pencegahan Tembus Batas Menggunakan Soft-Collision Dinding)
    // Menciptakan zona penahan elastis setebal radius h di sekeliling dinding interior kontainer.
    // ==============================================================================
    float wall_stiffness = 30.f; 
    
    float dist_left   = my_pos.x - container_left;
    float dist_right  = container_right - my_pos.x;
    float dist_bottom = container_bottom - my_pos.y;

    // Hukum Hooke Buatan: Gaya tolak berbanding lurus dengan kedalaman penetrasi partikel ke dinding
    if (dist_left < h)   velocities[id].x += (h - dist_left) * wall_stiffness * deltaTime;
    if (dist_right < h)  velocities[id].x -= (h - dist_right) * wall_stiffness * deltaTime;
    if (dist_bottom < h) velocities[id].y -= (h - dist_bottom) * wall_stiffness * deltaTime;

    // ==============================================================================
    // 2. LOGIKA UTAMA: SPATIAL HASHING NEIGHBOR SEARCH
    // Menghitung interaksi lokal antarpartikel fluida SPH.
    // ==============================================================================
    // Proyeksi koordinat partikel kontinu ke indeks grid 2D tempat dirinya berada
    int my_cell_x = (int)((my_pos.x - container_left) / cellSize);
    int my_cell_y = (int)((my_pos.y - container_top) / cellSize);

    // Iterasi 9-Sektor Spasial: Memeriksa sel tempat partikel berada + 8 sel tetangga di sekelilingnya
    for (int offset_y = -1; offset_y <= 1; ++offset_y) {
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            int target_cell_x = my_cell_x + offset_x;
            int target_cell_y = my_cell_y + offset_y;

            // Pastikan koordinat sel target valid berada di dalam area batas grid dunia
            if (target_cell_x >= 0 && target_cell_x < gridWidth && 
                target_cell_y >= 0 && target_cell_y < gridHeight) 
            {
                // Konversi koordinat matriks 2D ke ID Sel linear 1D (Tabel Hash Spasial)
                int target_cell_id = target_cell_y * gridWidth + target_cell_x;
                
                // Ekstraksi penunjuk indeks memori global terurut hasil Counting Sort dari CPU
                int start_idx = cellStart[target_cell_id];
                int count = cellCount[target_cell_id];

                // Looping hanya pada partikel-partikel j yang terdaftar di dalam bucket sel spasial ini
                for (int j = 0; j < count; ++j) {
                    int neighbor_id = start_idx + j;
                    
                    // Hindari perhitungan interaksi dengan dirinya sendiri (Self-Interaction)
                    if (neighbor_id == id) continue;

                    // Mengambil posisi tetangga j dari global memory
                    float2 neighbor_pos = positions[neighbor_id];
                    float dx = my_pos.x - neighbor_pos.x;
                    float dy = my_pos.y - neighbor_pos.y;
                    
                    // Komputasi jarak kuadrat (Lebih hemat performa, menunda pemanggilan fungsi akar sqrt)
                    float dist_sq = (dx * dx) + (dy * dy);

                    // Validasi jarak: Apakah tetangga j masuk ke dalam radius jangkauan bola pengaruh h?
                    if (dist_sq < h * h && dist_sq > 0.0001f) {
                        float dist = sqrt(dist_sq);
                        
                        // Menghitung besaran gaya interaksi hidrostatik via fungsi pembantu kernel
                        float forceMagnitude = calculate_smoothing_force(dist, h, stiffness); 
                        
                        // Proyeksi arah vektor gaya radial menjauh secara merata dari pusat massa tetangga j
                        float2 forceDir = (float2)(dx / dist, dy / dist);
                        
                        // Akumulasikan perubahan kecepatan langsung ke array penampung (Adveksi Kecepatan)
                        velocities[id] += forceDir * forceMagnitude * deltaTime;
                    }
                }
            }
        }
    }
}

// ==============================================================================
// KERNEL HARDWARE 2: ADVEKSI POSISI & POSITIONAL COLLISION SOLVER
// ==============================================================================
/**
 * Thread GPU mengeksekusi integrasi waktu kinematika posisi (Symplectic Euler) 
 * sekaligus mengunci koordinat partikel secara absolut agar tidak menembus wadah kontainer.
 */
__kernel void update_posisi(__global float2* positions,
                            __global float2* velocities,
                            const float container_left,
                            const float container_right,
                            const float container_bottom,
                            const float deltaTime,
                            const int numParticles) 
{
    // Dapatkan indeks linear global yang unik untuk thread GPU saat ini
    int id = get_global_id(0);
    if (id >= numParticles) return; // Proteksi keluar indeks

    // Integrasi Waktu: Memperbarui koordinat posisi 2D spasial berdasarkan resultan kecepatan terbaru
    positions[id].x += velocities[id].x * deltaTime;
    positions[id].y += velocities[id].y * deltaTime;
    
    // Parameter geometri fisik butiran partikel
    float particle_radius = 4.0f; 
    float bounce = -0.6f;         // Koefisien restitusi pembalan dinding fisik (60% energi kinetik dipertahankan)

    // Pre-kalkulasi batas toleransi koordinat berdasarkan radius geometri partikel (Mencegah eror -11 OpenCL)
    float bound_left   = container_left + particle_radius;
    float bound_right  = container_right - particle_radius;
    float bound_bottom = container_bottom - particle_radius;
    
    // ------------------------------------------------------------------------------
    // IMPLEMENTASI CONTINUOUS COLLISION DETECTION (METODE PENALARAN REFLEKSI CERMIN)
    // ------------------------------------------------=============================
    // Jika partikel melompati dinding batas kontainer akibat deltatime yang renggang, 
    // koordinatnya akan dipantulkan secara simetris menggunakan rumus `(2 * batas) - posisi`.
    
    // 1. Validasi Tabrakan Dinding Kiri
    if (positions[id].x < bound_left) {
        positions[id].x = (2.0f * bound_left) - positions[id].x; 
        if (velocities[id].x < 0.0f) velocities[id].x *= bounce; // Balikkan arah vektor kecepatan X
    }
    // 2. Validasi Tabrakan Dinding Kanan
    else if (positions[id].x > bound_right) {
        positions[id].x = (2.0f * bound_right) - positions[id].x; 
        if (velocities[id].x > 0.0f) velocities[id].x *= bounce; // Balikkan arah vektor kecepatan X
    }

    // 3. Validasi Tabrakan Dinding Dasar (Lantai Fisik Kontainer)
    if (positions[id].y > bound_bottom) {
        positions[id].y = (2.0f * bound_bottom) - positions[id].y; 
        if (velocities[id].y > 0.0f) velocities[id].y *= bounce; // Balikkan arah vektor kecepatan Y
    }
}