function Hp = Compute_Entropy(p)
	if p~= 0
		Hp = -p * log2(p);
	else
		Hp = 0;
	end
end