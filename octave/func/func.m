#!/usr/bin/octave -qf
# Plots floor function for a given range

function [x, y] = sqrtAndSquare(a)
    x = sqrt(a)
    y = a^2
endfunction

[x,y] = sqrtAndSquare(64);
y/x
