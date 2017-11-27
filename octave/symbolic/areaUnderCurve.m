#!/usr/bin/octave -qf
# Finds the area under the curve

pkg load symbolic
#symbols

a = 0;
b = 2;

x = sym("x");
f = inline("x^2");

ezplot(f, [a, b])

[area, err, ev] = quad(f, a, b)

display("Area: "), disp(double(area))

pause;
input("Press any key to continue...")
