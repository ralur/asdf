things = zeros(1, 100);
for i = 1:100
    things(i) = (100/i)*(.982^i + (1-.982^i)*(i+1));
end