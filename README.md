This algorithm is based off of the K nearest neighbors (KNN) algorithm as described by by Agarwal et. al. in their 2021 paper "Causal Matrix Completion". You can play around with the values in lines 6–10, keeping in mind the limitation of $d$ having to be less than either $m$ and $n$.

You should see:
1. A graph showing how the badness score changes as K increases
2. A matrix of matrices showing the imputed matrix and stuff

You can comment out anything below line 130 as you prefer (for example if you want to see the imputed matrix with a certain K instead of the K that has the lowest badness score).

- - -
# Example
![K score/badness graph]("./images/K score badness.png")
![Imputed matrix](".images/Imputed matrix.png")
