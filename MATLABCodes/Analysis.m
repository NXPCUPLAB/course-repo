%Step 1: Set the parameters
m = 1000;
b = 50;
u = 500;

s = tf('s');
P_cruise = 1/(m*s+b)

step(u*P_cruise)

%%
%Step 2: To check the pole(s)
pzmap(P_cruise) 
axis([-1 1 -1 1])

%%
%Step 3: Check Bode Diagram
bode(P_cruise)