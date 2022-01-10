d=0.2;

coord = [
    0,   0,   0;
    d,   0,   0;
    d,   d,   0;
    0,   d,   0;
    0,   0,   d;
    d,   0,   d;
    d,   d,   d;
    0,   d,   d;
    2*d, 0,   0;
    2*d, d,   0;
    2*d, 0,   d;
    2*d, d,   d;
];

dof = [
    1,  2,  3;
    4,  5,  6;
    7,  8,  9;
    10, 11, 12;
    13, 14, 15;
    16, 17, 18;
    19, 20, 21;
    22, 23, 24;
    25, 26, 27;
    28, 29, 30;
    31, 32, 33;
    34, 35, 36;
];

edof = [
    1  1 2 3   4  5  6   7  8  9  10 11 12  13 14 15  16 17 18  19 20 21  22 23 24;
    2  4 5 6  25 26 27  28 29 30  7  8  9   16 17 18  31 32 33  34 35 36  19 20 21;
];


nnode = size(coord,1)
ndof = size(dof,1)*size(dof,2)
nel = size(edof,1)

[ex,ey,ez] = coordxtr(edof,coord,dof,8);

%ex = [0 d d 0 0 d d 0;
%      d 2*d 2*d d d 2*d 2*d d];

%ey = [0 0 d d 0 0 d d;
%      0 0 d d 0 0 d d];

%ez = [0 0 0 0 d d d d;
%      0 0 0 0 d d d d];



% Send data of undeformed geometry
%cfvv.solid3d.draw_geometry(edof,coord,dof,0.02,1)

ep = [2];

E = 210000000;
v = 0.3;
D = hooke(4,E,v);

K = zeros(ndof);
for i=(1:nel)
    Ke = soli8e(ex(i,:), ey(i,:), ez(i,:), ep, D);
    K = assem(edof(i,:), K, Ke);
end

f = zeros(ndof,1);
f(8) = -3000;
f(20) = 3000;

bc = [1 0; 2 0; 3 0; 4 0; 5 0; 6 0; 10 0; 11 0; 12 0; 13 0; 14 0; 15 0; 16 0; 17 0; 18 0; 22 0; 23 0; 24 0; 25 0; 26 0; 27 0; 28 0; 29 0; 30 0; 31 0; 32 0; 33 0; 34 0; 35 0; 36 0];
%bcPrescr = np.array([1,2,3,4,5,6,13,14,15,16,17,18])
a = solveq(K, f, bc);

ed = extract(edof,a);

es = zeros(ep*ep*ep,6,nel);
et = zeros(ep*ep*ep,6,nel);
eci = zeros(ep*ep*ep,3,nel);

for i=(1:nel)
    [es(:,:,i),et(:,:,i),eci(:,:,i)] = soli8s(ex(i,:),ey(i,:),ez(i,:),ep,D,ed(i,:));
    %es(:,:,i) = soli8s(ex(i,:),ey(i,:),ez(i,:),ep,D,ed(i,:));
end

%es
%et
%eci



save('3Dsolid.mat','coord','dof','edof','a','ed','es','et','eci')

%cfvv.beam3d.draw_displaced_geometry(edof,coord,dof,a,def_scale=5)

%Start Calfem-vedo visualization
%cfvv.show_and_wait()