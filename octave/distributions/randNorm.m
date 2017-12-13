#!/usr/bin/octave -qf
# Plots a histogram for a sample of random numbers fetched from the Gaussian distribution

randNormal = randn(1, 10e3);

hold on;
hist(randNormal, 64);
hold off;

pause;
input("Press any key to continue...")
