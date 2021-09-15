m = 1000;
b = 50;
r = 10;

s = tf('s');
P_cruise = 1/(m*s + b);

Kp = 100; 
C = pid(Kp);

T = feedback(C*P_cruise,1)

t = 0:0.1:20;
step(r*T,t)
axis([0 20 0 10])

%%
%Here, we increased Kp to 5000. Compare the results.
Kp = 5000; 
C = pid(Kp);

T = feedback(C*P_cruise,1)

t = 0:0.1:20;
step(r*T,t)
axis([0 20 0 10])

%%
Kp = 800; %We set Kp to this value. You can experiment with different values and compare the results.
Ki = 40; %Now we add the integrator controller
C = pid(Kp,Ki);

T = feedback(C*P_cruise,1);

step(r*T,t)
axis([0 20 0 10])