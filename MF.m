function [phi_alpha_vector, phi_omega_vector] = MF(x, nAlphaTriang, nOmegaTriang, alpha_bounds, omega_bounds)
% Function that is used to the compute the membership of the state of the
% DC motor to each of the memberhsip functions that are defined in order to
% implement the Fuzzy Q-iteration algorithm.

    %System state
    state1 = x(1);
    state2 = x(2);
    
    % Define the cores of the membership functions
    alpha_triangles = linspace(alpha_bounds(1),alpha_bounds(2),  nAlphaTriang);
    omega_triangles = linspace(omega_bounds(1),omega_bounds(2),  nOmegaTriang);
    
    % Vector that contains the membership functions for alpha
    phi_alpha_init   = max([0,(alpha_triangles(1,2)-state1)/(alpha_triangles(1,2)-alpha_triangles(1,1))]);
    phi_alpha_stop   = max([0,(state1-alpha_triangles(1,nAlphaTriang-1))/(alpha_triangles(1,nAlphaTriang)-alpha_triangles(1,nAlphaTriang-1))]);
    phi_alpha_med    = max(zeros(nAlphaTriang-2,1),min(transpose((state1-alpha_triangles(1,(2:nAlphaTriang-1)-1))./(alpha_triangles(1,(2:nAlphaTriang-1))-alpha_triangles(1,(2:nAlphaTriang-1)-1))),transpose((alpha_triangles(1,(2:nAlphaTriang-1)+1)-state1)./(alpha_triangles(1,(2:nAlphaTriang-1)+1)-alpha_triangles(1,(2:nAlphaTriang-1))))));
    
    phi_alpha_vector = [phi_alpha_init; phi_alpha_med; phi_alpha_stop];
    
    % Vector that contains the membership functions for omega
    phi_omega_init   = max([0,(omega_triangles(1,2)-state2)/(omega_triangles(1,2)-omega_triangles(1,1))]);
    phi_omega_stop   = max([0,(state2-omega_triangles(1,nOmegaTriang-1))/(omega_triangles(1,nOmegaTriang)-omega_triangles(1,nOmegaTriang-1))]);
    phi_omega_med    = max(zeros(nOmegaTriang-2,1),min(transpose((state2-omega_triangles(1,(2:nOmegaTriang-1)-1))./(omega_triangles(1,(2:nOmegaTriang-1))-omega_triangles(1,(2:nOmegaTriang-1)-1))),transpose((omega_triangles(1,(2:nOmegaTriang-1)+1)-state2)./(omega_triangles(1,(2:nOmegaTriang-1)+1)-omega_triangles(1,(2:nOmegaTriang-1))))));
    
    phi_omega_vector = [phi_omega_init; phi_omega_med; phi_omega_stop];
end