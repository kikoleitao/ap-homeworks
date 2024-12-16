# Question 2

## 1) "Identify the learning rate that achieved the highest validation accuracy, reporting corresponding test and validation accuracy. Plot the training and validation losses for each configuration, and compare the differences between the validation loss curves, explaining these differences by relating to the learning rate values."

lr-0.00001:
val acc: 0.4694
test acc: 0.4623

lr-0.001:
val acc: 0.5264
test acc: 0.5247

lr-0.1:
val acc: 0.3689
test acc: 0.3843

The learning rate that achieved the highest validation accuracy was 0.001, yielding a validation accuracy of 0.5264 and a test accuracy of 0.5247.

Smallest Learning Rate (0.00001)
For the smallest learning rate of 0.00001, the training and validation losses decrease steadily but very slowly over 100 epochs. The small updates to the model parameters result in slow convergence. While this approach avoids overshooting or divergence, it would require significantly more epochs to achieve competitive performance compared to other models.

Highest Learning Rate (0.1)
In contrast, the model with the highest learning rate of 0.1 exhibits highly unstable behavior. The validation loss fluctuates significantly, reaching values over 10 times higher than those observed with the smallest learning rate, while the training loss stabilizes early at around 25. The large step sizes caused by the high learning rate prevent the model from settling near the optimal parameter values, leading to instability and poor generalization. Consequently, this model achieves the lowest validation and test accuracies of the three.

Optimal Learning Rate (0.001)
The learning rate of 0.001 strikes a balance between the two extremes. Both the training and validation losses decrease steadily and at an effective pace. The validation loss stabilizes at around 1.3, slightly higher than the training loss, without the instability seen in the highest learning rate or the slow convergence of the smallest learning rate.

Comparison and Conclusion
Among the three models, the learning rate of 0.001 demonstrates the best performance. It provides sufficient updates to the model parameters without diverging or fluctuating excessively. The steady decrease in validation loss indicates effective learning and convergence within 100 epochs, resulting in the highest validation and test accuracies.


## 2)

## a) "(a) (8 points) Train 2 models: one using the default hyperparameters and another with batch_size 512 and the remaining hyperparameters at their default value. Plot the train and validation losses for both, report the test and validation accuracy for both and explain the differences in both performance and time of execution."

batch_size-default64:
val acc: 0.5862
test acc: 0.5833
time: 2m 26s

batch_size-512:
val acc: 0.5085
test acc: 0.5347
time: 1m 39s


Comparison of Batch Sizes: 64 vs. 512
Analyzing the training and validation loss plots, we observe that the model with a batch size of 64 shows faster initial decreases in losses, stabilizing earlier compared to the model with a batch size of 512. This difference is even more pronounced in the validation accuracy plots. For instance, by epoch 25, the model with batch size 64 achieves a validation accuracy of nearly 50%, while the model with batch size 512 reaches only about 30%.

The numerical results further highlight this difference, with the model using batch size 64 showing approximately 15% improvement in validation accuracy and 10% improvement in test accuracy over the model with batch size 512.

These results can be attributed to the nature of smaller batch sizes. A smaller batch size provides more frequent updates to the model's weights, resulting in better gradient estimation and improved learning. However, this comes at the cost of increased training time. Smaller batch sizes require more iterations per epoch, as reflected in the results: the model with batch size 512 completes training 36 seconds faster than the one with batch size 64, representing a 31% reduction in training time.

Conclusion: Trade-offs in Batch Size
In conclusion, choosing a batch size involves a trade-off. Larger batch sizes offer faster training but may compromise generalization and accuracy, while smaller batch sizes improve performance and accuracy at the cost of longer training times. For this experiment, the smaller batch size of 64 delivers better results in terms of accuracy, though at a higher computational cost.



## b) (9 points) Train the model setting dropout to each value in {0.01, 0.25, 0.5} while keep ing all other hyperparameters at their default values. Report the final validation and test accuracies and plot the training and validation losses for the three configurations. Analyze and explain the results. 

TODO: ver melhor este again, foi o que os resultados mudaram mais
-> val acc > test acc para todos
-> observações dos gráficos não me pareceram mto accurate ja

dropout-0.01:
val acc: 0.5762
test acc: 0.5803

dropout-0.25:
val acc: 0.6083
test acc: 0.6057

dropout-0.5:
val acc: 0.5990
test acc: 0.5960

Analysis of Dropout Rates: 0.01, 0.25, and 0.5
At first glance, the training and validation loss plots for the three models appear similar. However, closer inspection reveals key differences worth explaining.

Smallest Dropout (0.01)
The model with the smallest dropout rate of 0.01 exhibits the lowest values of test and validation accuracy among the three, suggesting that minimal dropout leads to higher overfitting. This happens because with minimal regularization, the model closely fits the training data, overly relying on specific neurons, which limits its ability to generalize effectively.

Moderate Dropout (0.25)
The model with a dropout rate of 0.25 has better accuracy than the previous model, as a more substancial regularization prevents overfitting. We have a higher training loss which is balanced with a lower validation loss after stabilizing. 

Largest Dropout (0.5)
The model with 0.5 dropout rate doesn't improve relating to the previous one on both accuracies, indicating potential over-regularization, but still performs better than the first model with the lowest dropout. Again, the training loss increases in relation to the previous model, as it struggles to learn due to the high rate of dropout, but validation stabilizes at a similar value to the previous model.

Conclusion: Trade-offs in Dropout Rates
The trend we can see from the plots if we compare them in sequence, from smallest to largest dropout, is that the training loss keeps getting higher while the validation loss keeps settling on a lower value with more stability (without so many ups and downs).

The dropout rate of 0.25 provides the best tradeoff between reducing overfitting and maintaining model capacity, as seen in both accuracy metrics and the loss behavior.


## c) (8 points) Using a batch_size of 1024, train the default model while setting the momentum parameter to each value in {0.0; 0.9} (use the -momentum flag). For the two configurations, plot the train and validation losses and report the test and validation accuracies. Explain the differences in performance.


momentum-0.0:
val acc: 0.5862
test acc: 0.5833

momentum-0.9:
val acc: 0.6175
test acc: 0.6130

Analysis of Momentum: 0.0 vs. 0.9
Momentum-0.0 Model
The model with momentum set to 0.0 relies solely on the gradients from the current batch for updates. This leads to slower convergence and steady, but suboptimal, weight adjustments, making it more difficult for the model to escape local minima or saddle points. As a result, this model achieves lower validation and test accuracy compared to the model with momentum.

Momentum-0.9 Model
In contrast, the model with momentum set to 0.9 benefits from incorporating information from previous updates. This accelerates convergence and allows the optimizer to take more directed steps toward the optimal solution. Consequently, this model achieves higher validation and test accuracy, demonstrating its superior ability to optimize and generalize compared to the no-momentum model.

However, an interesting observation emerges from the validation loss plot of the momentum-0.9 model. After around epoch 50, the validation loss begins to increase, even as the validation accuracy fluctuates between 60-65%. This divergence between training and validation performance indicates potential overfitting, where the model starts to capture noise and less relevant patterns in the training data.

Conclusion
Despite signs of overfitting in the momentum-0.9 model, it remains more effective at generalizing to unseen data than the momentum-0.0 model, as evidenced by its superior test accuracy. This underscores the advantage of using momentum in optimization to accelerate convergence and improve performance.

