# Question 2

## 1) "Identify the learning rate that achieved the highest validation accuracy, reporting corresponding test and validation accuracy. Plot the training and validation losses for each configuration, and compare the differences between the validation loss curves, explaining these differences by relating to the learning rate values."

TODO: voltar a correr se for para fazer o forward sem activation, texto deve manter-se igual (sexta)

lr-0.00001:
val acc = 0.2963
test acc = 0.3137

lr-0.001:
val acc = 0.5114
test acc = 0.5127

lr-0.1:
val acc = 0.4886 (maximum val acc of 0.5107 in training epoch 16)
test acc = 0.4923

The learning rate that achieved the highest validation accuracy was 0.001, having achieved a validation accuracy of 0.5114 and test accuracy of 0.5127.

Starting with the training and validation losses plot for the smallest learning rate, 0.00001, we verify that both training and validation losses decrease very slowly over the 100 epochs. Since the learning rate is so small, the updates are also small, leading to a slow convergence. While this model avoids overshooting the loss and avoiding diverging, it would require significantly more epochs to achieve a competitive performance to the other models.

Moving on to the plot with the highest learning rate, 0.1, we can see that the validation loss fluctuates significantly while the training loss stabilizes early in the training at around 1.52, whcih suggests a degree of overfitting. The high learning rate causes large updates to the model parameters, creating a different problem from the model above. While the validation lost descreases fast, the large step size makes it difficult for the model to settle near the optimal paramater values, making the validation lost bounce up and down, which translates to high instability and low generalization capacity to the validation set.

Having seen the problems with both high and low learning rates, we can now look at the plot for the learning rate of 0.001, which as said before achieved the highest validation accuracy of 0.5114. In this plot we verify that both training and validation losses decrease steadily, not havint he instability problems of the highest learning rate, nor the poor performance form the smallest learning rate. Comparing the three models, this one achieves a good balance between sufficient updates to weights while avoiding diverging. The steady decrease in validation loss without much fluctuation indicates effective learning and convergence within the 100 epochs.


## 1)

## a) "(a) (8 points) Train 2 models: one using the default hyperparameters and another with batch_size 512 and the remaining hyperparameters at their default value. Plot the train and validation losses for both, report the test and validation accuracy for both and explain the differences in both performance and time of execution."

TODO: explicar porque é que no gráfico batch-64 a validation loss estabiliza mais acima da train loss
    -> Dúvida: não é overfitting no fundo? Validation loss estabiliza enquanto a training loss diminui

batch_size-default64:
val acc = 0.5848
test acc = 0.5863
time = 1m 56s

batch_size-512:
val acc = 0.5078
test acc = 0.5320
time = 1m 20s

Given these results, we have an improvement of ~15% in the validation accuracy, ~10% in testing accuracy from the model with batch size 512 to the one with batch size 64. This improvement can be explained due to the fact that a smaller batch size provides more frequent updates to the weights of the model, leading to a better estimation of the gradient. However, this comest at the cost of increased training time, since we need to have more iterations per epoch for the smaller batch size. This diffence is also reflected in the results where the model with batch size 512 takes 36 seconds less than the one with batch size 64, having a reduction of ~31% in training time. Concluding, we have a clear trade-off: larger batch sizes offer faster training but may sacrificy generalization and accuracy, while smaller batch sizes have improved performance at the cost of increased training time.



## b) (9 points) Train the model setting dropout to each value in {0.01, 0.25, 0.5} while keep ing all other hyperparameters at their default values. Report the final validation and test accuracies and plot the training and validation losses for the three configurations. Analyze and explain the results. 

TODO: fazer, resultados ja prontos




## c) (8 points) Using a batch_size of 1024, train the default model while setting the momentum parameter to each value in {0.0; 0.9} (use the -momentum flag). For the two configurations, plot the train and validation losses and report the test and validation accuracies. Explain the differences in performance.

TODO: fazer, resultados ja prontos
