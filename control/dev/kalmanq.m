syms q pp c pm
syms a1
syms r positives
A = [[a1, -1];[1, 0]];
P = [[pm, 0];[0, pp]];
Q = [[q, 0];[0, 0]];
H = [1,0];
R = [r];
AT = [[a1, 1];[-1, 0]];
HT = [[1];[0]];
disp('Setup done')
P_ricatti = A*P*AT + Q - (A*P*HT/(H*P*HT + R)*H*P*AT);
approx_P_ricatti = taylor(P_ricatti, q, 'Order', 2);
solution = solve([P(1);P(4)] == [approx_P_ricatti(1);approx_P_ricatti(4)], pm, pp);
pm_ricatti = solution.pm(1); % i hope this is the positive one
% solution = solve(P == taylor(P_ricatti, q, 'Order', 2), pm, pp);
disp('System set up')
K = [[pm_ricatti/(pm_ricatti + R(1))];[0]];
Kprime = diff(K, q);
toexp = A - K*H;
[S,D] = eig(toexp);
disp('A - KH diagonalized')
d1 = D(1);
d2 = D(4);
leftsum = [[(1)/(1 - d1), 0];[0, (1)/(1 - d2)]];
leftsum = S * leftsum/S;
left = H * leftsum * Kprime;
disp('Left set up')
rightsum = [[1/(1 - d1)^2, 0];[0, 1/(1 - d2)^2]];
rightsum = S * rightsum/S;
right = H * rightsum * K * H * Kprime;
disp('Right set up')
disp(left)
approxleft = taylor(left, q, 'Order', 2)
approxright = taylor(right, q, 'Order', 2)
fullsolution = solve(taylor(left, q, 'Order', 2) == taylor(right, q, 'Order', 2), q)