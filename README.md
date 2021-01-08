# Attacks-on-Machine-Learning-Models-Using-Adversarial-Examples

Summary:                
This project explores attacks and the quality of adversarial examples that they create. The implementation of the attacks is provided by the Adversarial Robustness 360 Toolbox(ART). Below you can find a list of the implemented attacks and their respective papers.
* Threshold Attack Vargas et al., 2019
* Pixel Attack Vargas et al., 2019; Su et al., 2019
* HopSkipJump attack Chen et al., 2019
* Projected gradient descent Madry et al., 2017
* NewtonFool Jang et al., 2017
* Elastic net attack Chen et al., 2017
* Spatial transformation attack Engstrom et al., 2017
* Query-efficient black-box attack Ilyas et al., 2017
* Zeroth-order optimization attack Chen et al., 2017
* Decision-based attack / Boundary attack Bren- del et al., 2018
* Adversarial patch Brown et al., 2017        
* Carlini & Wagner (C&W) L2 and L∞ attacks        
* Carlini and Wagner, 2016
* Basic iterative method Kurakin et al., 2016
* Jacobian saliency map Papernot et al., 2016
* Universal perturbation Moosavi-Dezfooli et al., 2016        
* DeepFool Moosavi-Dezfooli et al., 2015
* Virtual adversarial method Miyato et al., 2015
* Fast gradient method Goodfellow et al., 2014                                 
The victim model for this project is a pretrained model from TensorFlow hub. The specific model can be found here: https://tfhub.dev/deepmind/ganeval-cifar10-convnet/. To load the model and use it you can check out their tutorial on the subject https://www.tensorflow.org/hub/tf2_saved_model. The dataset is also available from either Tensorflow (https://www.tensorflow.org/datasets/catalog/cifar10 ) or Keras (https://keras.io/datasets/). 
        
Metrics: 
        
We chose to use classification_report from the Scikit-Learn library, which includes a confusion matrix, Recall, Precision, F1-score and support. In addition to that, we chose to include accuracy. As it is known in the field of Data Science, accuracy is not always the best metric to use when measuring the performance of models. Accuracy is strongly dependent on the underlying nature of the data. For example, let’s imagine a binary classifier that is to perform on a dataset that contains 100 points and 98 of those points belong to the positive class. Even if the classifier only predicted positive labels, it would be correct 98 out of 100 times, which would yield a score of 98%. Based on the description alone, our classifier is not a good one, actually it is not even a classifier at all because it gives the same result regardless of the input. Yet, its accuracy score is 98%. 


To address this, we rely on other metrics to evaluate the performance of our classifiers. 
Among these are the ones mentioned above, which are included in the classfication_report provided by Scikit-Learn. So for every model evaluation, which is done before and after each attack and/or defense is applied, we run the classification report with parameters true labels and predicted labels, respectively. The first output of the function is a confusion matrix. A confusion matrix, itself, can be thought of as a super-metric because it provides most of the information needed to compute the other metrics. It also gives us direct insight into the predictive patterns of our model because for each class it shows the number of correct predictions for the given class accompanied by exact numbers for all mislabeled examples and the class for which they were misclassified. A brief explanation of each metric is defined below. 


* Accuracy- The amount that was predicted correctly over the entire dataset that was evaluated. In this case, because our test dataset is balanced, accuracy can be considered a reliable metric.


* Precision - For a given class, this represents the number of correct predictions over all the examples that were classified as said class. This is important because if the model classifies some classes more accurately than others, we would be able to gather such information from this metric. 


* Recall - The number of correctly predicted labels for a specific class over the amount of examples of that class in the dataset. As precision measures the models ability to predict a specific class, it can be said that recall shows its inability to correctly predict a specific class. It can be thought of as a class specific accurate score.


* F1-Score - For times when we need more insight on the relationship between recall and precision, we turn to the F1-score because it is the harmonic mean of precision and recall. The higher this number is for a given class, the better performance our model has for that class. As an example, let’s imagine that we have a model that makes 10 predictions for a given class and all 10 are correct. Yet, there are 100 examples belonging to that class in our dataset. This would mean we have a precision score of 1 but recall score of only 0.1. Our model can look really good or really bad, depending on which metric we choose. The F1-score resolves this by taking an average of the two, which would give us a score that  provides a more accurate representation of our models performance on examples that belong to the class. 


* Support - This represents the number of occurrences of each class in the dataset. Although support is included in the report, it does not add any additional information in our case because we make sure that our dataset is balanced before we perform any evaluation. 




ISSUES WITH ATTACKS AND DEFENSES


Defensese:
1. First issue is when it comes to using PixelDefend, and this doesn't happen with any of the other ones, if there is an exception thrown, even if the code is fixed it will still throw the same exception. I usually resolve this by rerunning the code cell that compiles the model and the one that initializes the classifier. If anyone has any idea on how to fix this please let me know. 


2. Pixel Defend has no attribute pixel_cnn. Attempted fix is to pass the classifier as an argument to that parameter. I have also tried using a PixelCNN model found in tensorflow_probability.distributions and faced the same upcoming error.


3. The error says that the get_activations() function, which is called within the pixel_defend.py file is missing required positional argument batch_size. Lo and behold, in line 96 of the pixel_defend.py file it does call the function without providing the required argument. So I'm not really sure how to get around this.


4. My other issue concerning the defenses is that it seems that for class labels, it affects the performance of every classifier the same way and that's by dropping its accuracy to .1. I'm pretty sure it might be something that I'm missing but I just don't know what that is.


5. High Confidence and Label smoothing sometimes seem to have no effect on the performance of the classifier. Once again, I might be missing something but I'm not sure what it is as of now.


        




Attacks
1. Confidence Low Uncertainty attack - throws error TypeError: Model must be a GPy Gaussian Process classifier!


2. Decision Tree attack throws error: Must be a decision tree based classifier


Part 2: Attack on Text Data


In this section we show that adversarial examples are not limited to image data. We recreate the attack proposed by Papernot et al. in their 2016 paper, Crafting Adversarial Input Sequences for Recurrent Neural Networks and use it on the IMDB movie review dataset. Our experiment was conducted using a jupyter notebook, using Google Colab as a host. The experimental set up is as follows:


1. Data
We start by importing our dataset through the tensorflow.keras.datasets api. When the data is loaded, it returns two tuples which comes in the form (training examples, training label) and (testing examples, testing label). Because our model is expecting a fixed number of features and given that each review is a sequence of words(features), we set our maximum limit for reviews to 80 words and we pad the sequences to make sure that they are all the same length. We then use Scikit-Learn’s train_test_split function to get a balanced subset of our dataset, containing 2000 training examples and 100 testing examples. 


        For better evaluation of our attack, we also import the word dictionary provided by the IMDB dataset, which is a python dictionary in the form of word: index, for each key and value pair. Since the dataset is provided as sequences of indexes, we invert the dictionary to index:word format for easy look-up using the indexes. This gives us a way to, not only show sentence representation of a sequence, but also which words are substituted in each sentence as adversarial examples are being created. 
        
2. Model
For our model we use the suggested model mentioned in [2] as a foundation. However, we use a dense layer as our last layer instead of the softmax layer and we remove the average pooling layer. Our model performed described in [2] so we felt no need to alter our architecture any further. We compile our model using binary-cross-entropy as the loss function, Adam as the optimizer and accuracy as the metric. 


  



We only use three epochs to train our model as it reaches the performance described in [2] by the second epoch. 


3. Attack Creation
As a first step to implementing the attack, we create a function, jacobian_for, that  takes as parameter an example and, using tensorflow’s GradientTape api, calculates the jacobian of the output of the model with respect to the input. We also create a convenient function that turns labels into ones or zeros. This is an important step as the model yields a probability value, which can become problematic when using a while loop that is conditioned on the model’s prediction on the adversarial examples being different from the original example. If this step is skipped, it would be the case that an adversarial example is successfully created but the loop would not stop executing.


After the above steps are complete, we create the function that actually creates the attack. The function takes as parameter, as described by [2], a model f, an example x, and a dictionary D. We also include an extra parameter max_iter, which determines how many times we want the while loop to iterate successfully creating an adversarial example. Following the algorithm described by [2] and shown below, we implement the attack. 
  



4. Attack Evaluation


Using the default value of 100 for max_iter, we were able to generate over 50 adversarial examples out of 100 test samples using an average of 35.4 iterations. This is simply an average. There are several adversarial examples that are created in as little as one or two iterations. It seems that some examples need significantly more iterations to converge. Some factors that play a part in how the model performs include the fact that the attack uses randomness to select which word to alter. Naturally, the limitations of the random number generator ends up playing a major role in the amount of iterations it takes before an adversarial example can successfully be created. And as this random number generator was used without an ever-changing seed, it can be assumed that after a number of iterations, it starts to repeat the same sequence of numbers.  When we tested our model on our generated adversarial examples, we were able to drop the accuracy from 82.9 percent to 66 percent. 


Along with the performance metrics provided above, we also print out the sentences before and after their alteration by the attack algorithm. Below we show a few samples from our subset of the dataset. 


Sentence  0 Original
credit  half  film  it  is  worn  over  genre  for  incidental  in  political  mafia  in  while  characters  not  an  that  end  it  cannot  of  self  slow  virginia  some  br  read  been  impressed  since  film  really  from  after  one  cinema  to  plays  is  now  on  then  also  we  enjoy  that  with  very  in  can  when  legs  from  off  ever  not  what  from  after  one  out  bit  up  film  of  shepherd  i  i  seen  mean  funny  very  less  half  scheming  this  of  and  


Sentence  0 Adversarial
credit  half  film  it  is  worn  already  genre  for  incidental  in  political  mafia  in  while  characters  not  an  that  end  it  cannot  of  self  slow  virginia  some  summoned  read  been  to  since  laughable  really  from  after  one  cinema  to  plays  is  now  on  then  also  matter  enjoy  that  with  very  in  can  when  legs  from  off  ever  not  what  from  after  one  out  bit  up  film  of  shepherd  i  i  seen  mean  already  very  less  half  scheming  this  of  and

Sentence  1 Original
i  have  well  can  its  br  of  little  experience  or  is  morality  tv  to  of  i  i  finally  it  mightily  film  do  by  source  to  own  and  not  was  tom  of  i  i  revenge  than  some  in  sister  options  of  weary  scene  her  released  movie  is  i  i  famous  without  brought  genre  that  trying  some  mexican  main  wife  are  i  i  8  left  final  wonderful  i  i  supporting  didn't  his  main  if  arnold  this  blame  not  find  but  mind  and  

Sentence  1 Adversarial
i  have  well  can  its  br  of  little  experience  or  is  morality  tv  to  of  i  i  finally  it  mightily  film  do  by  of  to  own  and  not  was  tom  of  i  i  revenge  than  some  in  sister  options  of  weary  scene  her  released  movie  is  i  i  famous  without  brought  genre  that  trying  some  mexican  main  wife  are  i  i  8  left  final  wonderful  i  i  supporting  didn't  his  main  if  arnold  this  blame  not  find  but  mind  and  


Sentence  2 Original
really  by  well  can  are  holocaust  distribution  oh  any  of  where  gray  this  of  young  that  into  at  done  in  can  what  as  going  david  twelve  it  movie  equally  movie  be  toned  into  sex  die  this  anti  rushed  good  my  board  think  very  you  link  had  rock  in  also  is  few  they  as  i  i  interesting  world  of  times  apparently  for  somehow  sum  this  of  construction  br  live  it  peters  want  in  sum  this  of  construction  br  an  br  want  


Sentence  2 Adversarial
really  by  matter  can  are  holocaust  distribution  oh  any  of  where  gray  this  of  young  that  into  at  done  summoned  can  what  as  going  david  twelve  it  movie  people  movie  be  toned  into  laugh  die  this  anti  rushed  moment  my  board  becomes  very  you  link  had  rock  in  also  is  few  they  as  i  jake's  interesting  world  of  times  apparently  caroline  somehow  to  this  of  to  br  live  summoned  peters  already  summoned  sum  to  of  construction  br  an  br  want 
Helpful links:
https://deepnotes.io/adversarial-attack   
https://keras.io/examples/imdb_lstm/ 
https://medium.com/unit8-machine-learning-publication/computing-the-jacobian-matrix-of-a-neural-network-in-python-4f162e5db180
