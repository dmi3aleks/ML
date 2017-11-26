#!/usr/bin/octave -qf
# Finds linear regression parameters for a given dataset

data = load('price_08:40AMSeptember062017.csv');
x = data(:,1);
y = data(:,2);

disp(x);
disp(y);
plot(x,y);
xlabel('time')
ylabel('price')


m = length(x);
X = [ones(m,1) x];
z = (pinv(X'*X))*X'*y
disp(z)

hold on; % plot a fitted line on the fame figure
plot(X(:,2),X*z,'-')
hold off;

pause;
input("Press any key to continue...")
