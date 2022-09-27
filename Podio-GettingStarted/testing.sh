for i in {1..10}
do
	./cc ../rec_profile_DigiReco_1k.root > alg_test.out
	sort alg_test.out > sorted_alg_test
	diff --color -s sorted_alg_cntrl sorted_alg_test
	if [[ $? -eq 1 ]]
	then
		echo "Test Output differs from Control at iteration $i!!\n"
	fi
done
