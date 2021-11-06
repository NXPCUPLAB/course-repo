m = 1000;
b = 50;
u = 500;

s = tf('s');
P_cruise = 1/(m*s+b);

Ts = 1/50;

dP_cruise = c2d(P_cruise,Ts,'zoh')

%%
%To check the root locus in the z-plane
Wn = 0.0072;
zeta = 0.6;

rlocus(dP_cruise)
zgrid(zeta, Wn)
axis ([-1 1 -1 1])

%%
%You can use axis ([0.95 1 -.1 .1]) to narrow down the graph to the area of
%interest
rlocus(dP_cruise)
zgrid(zeta, Wn)
axis ([0.95 1 -.1 .1])

%%
%Specify the desired root locus to get a specific value of K for the controller 
rlocus(dP_cruise)
zgrid(zeta, Wn)
axis ([-1 1 -1 1])
[K,poles] = rlocfind(dP_cruise)

%%
%You will get a value for K
K = 0; %Enter the value of the obtained K.
sys_cl = feedback(K*dP_cruise,1);
r = 10;
figure
step(r*sys_cl,10);