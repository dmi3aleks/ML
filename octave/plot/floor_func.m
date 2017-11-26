#!/usr/bin/octave -qf
# Plots floor function for a given range

function floor_plot(from, to, samples)
    x = linspace(from, to, samples);
    y = floor(x);
    plot(x,y);
endfunction

floor_plot(0, 42, 100)

input("Press any key to continue...")
