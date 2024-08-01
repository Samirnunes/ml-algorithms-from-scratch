# Self-Organizing Map (SOM)

A self-organizing map (SOM) is an unsupervised machine learning algorithm used to produce a low-dimensional (typically two-dimensional) representation of a higher-dimensional data set while preserving the topological structure of the data. Besides, an SOM is a type of artificial neural network but is trained using competitive learning rather than the error-correction learning. It was introduced by the Finnish professor Teuvo Kohonen in the 1980s and therefore is sometimes called a Kohonen map or Kohonen network [1].

The algorithm's training uses an input data set, in the input space, to generate a lower-dimensional representation of the input data, in the map space. Then, it's possible to use the generated map to classify additional input data in the respective group. Each group is called neuron and all groups are aranged in the map space generally using a rectangular grid. To relate each neuron to the input space, each one is associated with a weight vector, which is the position of the node in the input space. Each weight vector, in this implementation, is initialized with a random selected example from the training data. In this context, while nodes in the map space stay fixed, training consists in moving weight vectors toward the input data while reducing the Euclidean distance between the input and the neurons based in the initial distance between then, without spoiling the topological structure from the map space. 

## Data

Data for Admission in the University from Kaggle: https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university?source=post_page-----b3cdb9de1a24--------------------------------

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/linear_regression/images/data.png" alt="Material Bread logo">
<p>

## Implementation

The implementation is present in the file `som.py`, in the folder `src`. The theory used for that is based in the references [1] and [2], which can be found in the `references` folder.

When a training example fed to the network, its Euclidean distance to all weight vectors is computed. In this implementation, the training example is selected randomly from the dataset. The neuron whose weight vector is most similar to the input (smaller Euclidean distance) is called the best matching unit (BMU), and the weights of the BMU and neurons close to it in the SOM grid are adjusted towards the input vector. The magnitude of the change decreases between iterations and with the grid-distance from the BMU. The update formula for a neuron v with weight vector Wv(s) is:

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/self_organizing_map/images/update_formula.png" alt="Material Bread logo">
<p>

where s is the step index, t is an index into the training sample, u is the index of the BMU for the input vector D(t), α(s) is a monotonically decreasing learning rate; θ(u, v, s) is the neighborhood function which gives the distance between the neuron u and the neuron v in step s. 

In this implementation, a gaussian neighborhood function [2] is used:

<p align="center">
    <img width="200" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/self_organizing_map/images/gaussian_neighborhood_function.png" alt="Material Bread logo">
<p>

where σt is called the neighborhood radius, which represents how much other neurons are influenced by the BMU, and ||rc - ri|| is the distance between a neuron and the BMU.

Besides, the learning rate α and the neighborhood radius σt both decreases exponencially between iterations, according to the decay rate hyperparameter. 

In summary, the implemented SOM algorithm is the following one [1]:

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/self_organizing_map/images/algorithm.png" alt="Material Bread logo">
<p>

One already made implementation of SOM can be seen in the reference [3].

### Tools

- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Results

- SOM representation with neurons' weights (input space) and neurons' positions in the grid (map space):

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/self_organizing_map/images/som_map.png" alt="Material Bread logo">
<p>

- SOM pairplot after applying PCA (Principal Component Analysis) in the input points:

<p align="center">
    <img width="900" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/self_organizing_map/images/pairplot.png" alt="Material Bread logo">
<p>

- SOM scatterplot after applying PCA (Principal Component Analysis) in the input points:

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/self_organizing_map/images/scatterplot.png" alt="Material Bread logo">
<p>

- CGPA feature histogram by neuron:

<p align="center">
    <img width="900" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/self_organizing_map/images/cgpa_feature_hist.png" alt="Material Bread logo">
<p>

- University Rating feature histogram by neuron:

<p align="center">
    <img width="900" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/self_organizing_map/images/university_rating_feature_hist.png" alt="Material Bread logo">
<p>

SOM in fact separated data from the Admission in the University dataset (considering only the features - chance of admit, the target for a supervised learning technique, is dropped) in groups, as can be seen in the pairplot and in the scatterplot, and data's topological structure is maintained, because, according to the histograms, nearby neurons have similar points and distant neurons have clearly different data. For example, neuron 0 has data assigned to it with higher values of CGPA and University Rating, while neuron 3, the farthest away from neuron 0 in the map space, has lower values of CGPA and University Rating. It can be concluded that there are 3 clusters, according to the plots, since neurons 1 and 2 have similar data.

## References

[1] https://en.wikipedia.org/wiki/Self-organizing_map#:~:text=A%20self-organizing%20map%20(SOM,topological%20structure%20of%20the%20data.

[2] https://lamfo-unb.github.io/2020/08/29/Self-organizing-maps/

[3] https://github.com/JustGlowing/minisom/blob/master/minisom.py
