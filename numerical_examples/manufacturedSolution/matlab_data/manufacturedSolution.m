clc
clear all
%% Define problem parameters
N=400; % Number of eigenfrequencies to be calculated
c_0 = 450; % Speed of sound (m/s)
L = 0.4; % length of the acoustic domain (m)
h = 0.1; % height of the acoustic domain (m)
%% Pure real Z cases
k0 = 1-i; %Initial guess of k for 
r_a = linspace(-10, 10, N);
for i=1:length(r_a)
    % First find the root for real Z.
    x(i)=fsolve(@(k) dispersion2d(k, r_a(i), L, h), k0);
    f_a(i) = c_0*x(i)/(2*pi);
    %Apply continuation
    k0=x(i)
end
%% Pure imaginary Z cases
r_b = linspace(-10i, 10i, N);
k0 = 10; %Initial guess for k for pure imaginary Z
for i=1:length(r_b)
    % First find the root for imaginary Z.
    y(i)=fsolve(@(k) dispersion2d(k, r_b(i), L, h), k0);
    f_b(i) = c_0*y(i)/(2*pi);          
end
%% PLOT EIGENFREQUENCIES
figure;
sgtitle('Two Dimensional First Eigenmodes for Z=a+bi') 
subplot(2,2,1)
plot(imag(r_b),real(f_b))
xlabel('b')
ylabel('Re(f)')

subplot(2,2,2)
plot(imag(r_b),imag(f_b))
ylim([-1 1])
legend('Analytical')
xlabel('b')
ylabel('Im(f)')

subplot(2,2,3)
plot(r_a,real(f_a))
xlabel('a')
ylabel('Re(f)')

subplot(2,2,4)
plot(r_a,imag(f_a))

xlabel('a')
ylabel('Im(f)')
% saveas(gcf,'two2mode.pdf')
%% WRITE DATA TO TXT FILE
file = fopen("analytical.txt",'w');
formatSpec = '%2.1f ';
for j=1:length(f_a)
    fprintf(file, formatSpec, real(f_b(j)));
    fprintf(file, formatSpec, imag(f_b(j)));
    fprintf(file, formatSpec, real(f_a(j)));
    fprintf(file, ('%2.1f \n'), imag(f_a(j)));
end
fclose(file);