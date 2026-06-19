// File: coba.cl

float calculate_smoothing_force(float dist, float h, float stiffness) {
    if (dist < h && dist > 0.0001f) {
        float diff = h - dist;
        
        // --- TUNING BENTUK KURVA ---
        // Pilihan 1: Linear (Keras dan tajam)
        // return diff * stiffness; 
        
        // Pilihan 2: Kuadratik "Spiky" (Lebih natural untuk tekanan air, direkomendasikan)
        return pow(diff, 2) * stiffness; 

        // Pilihan 3: Kubik (Sangat halus, cocok untuk gas/asap)
        // return (diff * diff * diff) * stiffness;
    }
    return 0.0f;
}

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
    int id = get_global_id(0);
    if (id >= numParticles) return; // Keluar jika index melebihi jumlah partikel asli!

    float2 my_pos = positions[id];
    
    velocities[id].y += gravity * deltaTime;
    velocities[id] *= 0.99f; // Viskositas/Gesekan

    // ==============================================================================
    // FITUR INTERAKSI MOUSE
    // ==============================================================================
    if (mouseForce != 0.0f) {
        float m_dx = my_pos.x - mousePos.x;
        float m_dy = my_pos.y - mousePos.y;
        float m_dist_sq = (m_dx * m_dx) + (m_dy * m_dy);
        
        float mouseRadius = 100.0f; // Radius jangkauan efek mouse (Bisa diubah)

        // Jika partikel berada di dalam jangkauan radius mouse...
        if (m_dist_sq < mouseRadius * mouseRadius && m_dist_sq > 1.0f) {
            float m_dist = sqrt(m_dist_sq);

            float eps = 0.001f;
            
            // Semakin dekat dengan mouse, gayanya semakin kuat
            float forceMagnitude = (mouseRadius - m_dist) * mouseForce * deltaTime;
            
            // Arah gaya
            float dirX = m_dx / (m_dist + eps);
            float dirY = m_dy / (m_dist + eps);
            
            // Aplikasikan gaya. 
            // Jika Tarik (mouseForce positif), kecepatan dikurangi (mendekat ke mouse)
            // Jika Tolak (mouseForce negatif), kecepatan ditambah (menjauh dari mouse)
            velocities[id].x -= dirX * forceMagnitude;
            velocities[id].y -= dirY * forceMagnitude;
        }
    }

    // ==============================================================================
    // 1. SPH BOUNDARY COLLISION (Gaya Tolak Dinding)
    // ==============================================================================
    float wall_stiffness = 30.f; 
    
    float dist_left   = my_pos.x - container_left;
    float dist_right  = container_right - my_pos.x;
    float dist_bottom = container_bottom - my_pos.y;

    if (dist_left < h) velocities[id].x += (h - dist_left) * wall_stiffness * deltaTime;
    if (dist_right < h) velocities[id].x -= (h - dist_right) * wall_stiffness * deltaTime;
    if (dist_bottom < h) velocities[id].y -= (h - dist_bottom) * wall_stiffness * deltaTime;

    // ==============================================================================
    // 2. LOGIKA SPATIAL HASHING NEIGHBOR SEARCH
    // ==============================================================================
    int my_cell_x = (int)((my_pos.x - container_left) / cellSize);
    int my_cell_y = (int)((my_pos.y - container_top) / cellSize);

    for (int offset_y = -1; offset_y <= 1; ++offset_y) {
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            int target_cell_x = my_cell_x + offset_x;
            int target_cell_y = my_cell_y + offset_y;

            if (target_cell_x >= 0 && target_cell_x < gridWidth && 
                target_cell_y >= 0 && target_cell_y < gridHeight) 
            {
                int target_cell_id = target_cell_y * gridWidth + target_cell_x;
                int start_idx = cellStart[target_cell_id];
                int count = cellCount[target_cell_id];

                for (int j = 0; j < count; ++j) {
                    int neighbor_id = start_idx + j;
                    if (neighbor_id == id) continue;

                    float2 neighbor_pos = positions[neighbor_id];
                    float dx = my_pos.x - neighbor_pos.x;
                    float dy = my_pos.y - neighbor_pos.y;
                    float dist_sq = (dx * dx) + (dy * dy);

                    if (dist_sq < h * h && dist_sq > 0.0001f) {
                        float dist = sqrt(dist_sq);
                        
                        // PANGGIL FUNGSI SMOOTHING DI SINI!
                        float forceMagnitude = calculate_smoothing_force(dist, h, stiffness); 
                        
                        float2 forceDir = (float2)(dx / dist, dy / dist);
                        velocities[id] += forceDir * forceMagnitude * deltaTime;
                    }
                }
            }
        }
    }
}

// ==============================================================================
// KERNEL 2: UPDATE POSISI DAN TABRAKAN DINDING KOTAK
// ==============================================================================
__kernel void update_posisi(__global float2* positions,
                            __global float2* velocities,
                            const float container_left,
                            const float container_right,
                            const float container_bottom,
                            const float deltaTime,
                            const int numParticles) 
{

    int id = get_global_id(0);
    if (id >= numParticles) return; // Keluar jika index melebihi jumlah partikel asli!
    
    // Perbarui posisi berdasarkan kecepatan
    positions[id].x += velocities[id].x * deltaTime;
    positions[id].y += velocities[id].y * deltaTime;
    
    // Radius fisik partikel dan koefisien pantulan (Restitution)
    float particle_radius = 4.0f; 
    float bounce = -0.6f;         

    // DEKLARASIKAN SEMUA BATAS DI AWAL (Solusi Error -11)
    float bound_left   = container_left + particle_radius;
    float bound_right  = container_right - particle_radius;
    float bound_bottom = container_bottom - particle_radius;
    
    // 1. Tabrakan Dinding Kiri
    if (positions[id].x < bound_left) {
        positions[id].x = (2.0f * bound_left) - positions[id].x; 
        if (velocities[id].x < 0.0f) velocities[id].x *= bounce;
    }
    // 2. Tabrakan Dinding Kanan (Langsung menyambung dari if di atas)
    else if (positions[id].x > bound_right) {
        positions[id].x = (2.0f * bound_right) - positions[id].x; 
        if (velocities[id].x > 0.0f) velocities[id].x *= bounce;
    }

    // 3. Tabrakan Dinding Bawah (Lantai)
    if (positions[id].y > bound_bottom) {
        positions[id].y = (2.0f * bound_bottom) - positions[id].y; 
        if (velocities[id].y > 0.0f) velocities[id].y *= bounce;
    }
}