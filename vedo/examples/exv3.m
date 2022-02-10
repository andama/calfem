% CALFEM Vedo Visuaization example (exv3)
% Author: Andreas Åmand

clear
clc

d=0.1; % Elements are 100 mm x 100 mm x 100 mm 

% No. elements per direction
nel_x = 4;
nel_y = 10;
nel_z = 4;
nel = nel_x*nel_y*nel_z;

% No. nodes per direction
nnode_x = nel_x+1;
nnode_y = nel_y+1;
nnode_z = nel_z+1;
nnode = nnode_x*nnode_y*nnode_z;

% --- Creates Coord matrix ---

coord = zeros(nnode,3);
row = 1;
for z = 0:nnode_z-1
    for y = 0:nnode_y-1
        for x = 0:nnode_x-1
            coord(row,:) = [x*d,y*d,z*d];
            row = row+1;
        end
    end
end

% --- Creates Dof matrix ---

dof = zeros(nnode,1);
it = 1;
for row = 1:nnode
    dof(row) = it;
    it = it + 1;
end
ndof = size(dof,1)*size(dof,2);

% --- Creates Edof and Boundary condition matrices ---
% Boundary conditions: nodes at x = 0m & x = 50m have a displacement of 0

x_step = 1; % Next node in x-direction
y_step = nnode_x; % Next node in y-direction
z_step = (y_step)*nnode_y; % Next node in z-direction

it = 1; % Element number for loops (used as index in edof)
bc_it = 1; % Iteration for bc (used as index in bc)
force_dof_it = 1; % Iteration for point load dofs (used as index in force_dofs)
node = 1; % For keeping track of node

edof = zeros(nel_x*nel_y*nel_z,8+1);
bc = zeros(nnode_y*nnode_z*2,2);
force_dofs = zeros(25+1,1); % for saving dofs to apply point loads to
for col = 0:nel_z-1 % Loops through z-axis
    node = 1+z_step*col;
    for row = 0:nel_y-1 % Loops through y-axis
        for el = 0:nel_x-1 % Loops through x-axis
            edof(it,1) = it; % Element number, first row in Edof

            % --- First node ---
            edof(it,2) = dof(node,:); % Dofs for first element node
            if el == 0  % If element is at x = 0, save bc
                bc(bc_it,1) = dof(node,1);
                bc_it = bc_it + 1;
            end
            
            % --- Second node ---
            node = node+x_step; % Gets node number
            edof(it,3) = dof(node,:); % Gets dofs for node
            if el == nel_x-1 % If element is at x = 5, save bc
                bc(bc_it,1) = dof(node,1);
                bc_it = bc_it + 1;
            end
            
            % --- Third node ---
            node = node+y_step;
            edof(it,4) = dof(node,:);
            if el == nel_x-1
                bc(bc_it,1) = dof(node,1);
                bc_it = bc_it + 1;
            end

            % If elements at x = 0 and top row, save y-dofs for later
            %if (col == 0 && row == 3 && ismember(el,(13:38)) == 1)
            %    force_dofs(force_dof_it) = dof(node,2);
            %    force_dof_it = force_dof_it + 1;
            %end
            
            % --- Fourth node ---
            node = node-x_step;
            edof(it,5) = dof(node,:);
            if el == 0
                bc(bc_it,1) = dof(node,1);
                bc_it = bc_it + 1;
            end
            
            % --- Fifth node ---
            node = node+z_step-y_step;
            edof(it,6) = dof(node,:);
            if el == 0
                bc(bc_it,1) = dof(node,1);
                bc_it = bc_it + 1;
            end
            
            % --- Sixth node ---
            node = node+x_step;
            edof(it,7) = dof(node,:);
            if el == nel_x-1
                bc(bc_it,1) = dof(node,1);
                bc_it = bc_it + 1;
            end
            
            % --- Seventh node ---
            node = node+y_step;
            edof(it,8) = dof(node,:);
            if el == nel_x-1
                bc(bc_it,1) = dof(node,1);
                bc_it = bc_it + 1;
            end
            
            % --- Eighth node ---
            node = node-x_step;
            edof(it,9) = dof(node,:);
            if el == 0
                bc(bc_it,1) = dof(node,1);
                bc_it = bc_it + 1;
            end

            
            
            % Reset node
            if el == nel_x-1 % If last element
                node = node-z_step-y_step+2;
            else % Otherwise, first node for next el. = second node for current
                node = node+x_step-y_step-z_step;
            end
            
            it = it+1;

        end
    end
end

% --- Creating global Stiffness & Force matrices ---

[ex,ey,ez] = coordxtr(edof,coord,dof,8);

ep = [2]; % No. integration points

Lambda = 1.4; % Thermal conductivity for concrete
%D = ones(3,3)*Lambda;
D = eye(3)*Lambda;

eq = zeros(nel,1);
eq(11) = 10000;
eq(150) = 10000;

f = zeros(ndof,1);
K = zeros(ndof);
for i=(1:nel) % Assembling
    [Ke,fe] = flw3i8e(ex(i,:), ey(i,:), ez(i,:), ep, D, eq(i));
    [K,f] = assem(edof(i,:), K, Ke, f, fe);
end

% --- Solving system of equations ---

a = solveq(K, f, bc);

% --- Extracting global displacements ---

ed = extract(edof,a);

% --- Extracting global displacements & calculating element stresses ---

es1 = zeros(ep*ep*ep,3,nel);
et1 = zeros(ep*ep*ep,3,nel);
eci1 = zeros(ep*ep*ep,3,nel);
for i=(1:nel)
    [es1(:,:,i),et1(:,:,i),eci1(:,:,i)] = flw3i8s(ex(i,:),ey(i,:),ez(i,:),ep,D,ed(i,:));
end

es = zeros(nel,3);
et = zeros(nel,3);
eci = zeros(nel,3);
for i = (1:nel)
    es(i,1) = mean(es1(:,1,i));
    es(i,2) = mean(es1(:,2,i));
    es(i,3) = mean(es1(:,3,i));

    et(i,1) = mean(et1(:,1,i));
    et(i,2) = mean(et1(:,2,i));
    et(i,3) = mean(et1(:,3,i));

    eci(i,1) = mean(eci1(:,1,i));
    eci(i,2) = mean(eci1(:,2,i));
    eci(i,3) = mean(eci1(:,3,i));
end

save('exv3.mat','coord','dof','edof','bc','eq','a','ed','es','et','eci')
