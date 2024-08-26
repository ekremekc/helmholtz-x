function func = nicoud(k, Z, L, h)
    k_y = sqrt(k^2-((1*pi)/L)^2);
    func = exp(2i*k_y*h)*(k_y-k/Z)-(k_y+k/Z);
