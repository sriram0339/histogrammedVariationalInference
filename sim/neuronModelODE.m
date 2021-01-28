

x0 = [-1;1];

[times, x] = ode45(@fitzhughNagumo, [0:1:20], x0);

subplot(2,1,1);
plot(times, x(:,1));
subplot(2,1,2);
plot(times, x(:,2));


function dx = fitzhughNagumo(t, x) 
p1 = 0.3;
p2 = 0.1;
p3 = 0.5;
   x1 = x(1,1);
   x2 = x(2,1);
   
   dx1 = p3 * (x1 - x1^3/3 + x2);
   dx2 = -1/p3 * (x1 - p1 + p2 * x2);
   
   dx =[dx1;dx2];
   
end