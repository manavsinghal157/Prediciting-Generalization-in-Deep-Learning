
"""Implementation of CNA complexity measure"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
def complexity(model, dataset):
    def get_entropy(x):
        hist = np.histogram(x, bins=1000,density=True)
        data = hist[0]
        ent=0
        for i in hist[0]:
            if i!=0:
                ent -= i * np.log2(abs(i))
        return ent

    def get_entropy_batch_np(x):
#         x=np.array(x)
#         #x=np.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]))
#         x = tf.convert_to_tensor(x)
        entropy = []
        for i in range(x.shape[0]):
            entropy.append(get_entropy(x[i]))
        entropy = np.array(entropy)
        return entropy

    def get_slopes(x,model):
      activations = []
      for layer in model.layers: 
         x = layer(x) 
         activations.append(x)
      Temp1=[]
      for i in activations:
        j = np.reshape(i, (i.shape[0], -1))
        Temp1.append(np.array(np.sum(j, axis = 1)))
        Temp = np.array(Temp1)
      acts=np.reshape(Temp, (Temp.shape[0], Temp[0].shape[0]))
      def threshold(M):
          Mabs = np.abs(M)
          M[Mabs<0.0000001] = 0
          return M
      C = np.array([np.ones(len(acts)),np.arange(1,len(acts)+1)]).transpose()
      Cf = np.linalg.inv((C.T).dot(C)).dot(C.T)
      Cf = threshold(Cf)
      Cf = Cf[1,:]
      S = 0
      for j in range(len(Cf)):
          S += acts[j]*Cf[j]
      return S

    def get_CNA(dataset, model):
      #dataset = dataset.enumerate()
#      dataset = dataset.enumerate()
#      x=[]
#      for element in dataset.as_numpy_iterator():
#        print(element[1][0])
#        x.append(element[1][0])
#      x=np.array(x) 
      
#       #print(dataset.get_shape)
      coeff=[]
      for i, (x, y) in enumerate(dataset.batch(16)):
        entropy = get_entropy_batch_np(x)
        slopes = get_slopes(x,model)
        coeff.append(np.corrcoef(entropy,slopes)[0,1])
      coeff=np.array(coeff)  
           
      
# #       """for element in dataset.as_numpy_iterator():
# #         x.append(element[1]['image'])
# #       entropy = get_entropy_batch_np(x)
# #       slopes = get_slopes(x,model)"""
      return np.mean(coeff)

    return get_CNA(dataset,model)

