## Import Training Data From file
## Import Testing Data From file
import numpy as np
from matplotlib import pyplot as plt

with open('train-labels.idx1-ubyte', 'rb') as f:
    labels = f.read()
    labels = list(labels)
with open('train-images.idx3-ubyte', 'rb') as f:
    images = f.read()
with open('t10k-labels.idx1-ubyte', 'rb') as f:
    t_labels = f.read()
    t_labels = list(t_labels)
with open('t10k-images.idx3-ubyte', 'rb') as f:
    t_images = f.read()

## Remove the magic Number
def removeMGnum(labels,imgs):
  lbels = []
  img = []

  for i in range(len(labels)-8):
    lbels.append(labels[i+8])
    dum = []
    for m in range(28):
      
      for n in range(28):
        if imgs[m*28+n+28**2*i+16]==0:
          dum.append(0)
        else:
          dum.append(1)
    img.append(dum)
  return(lbels,img)

##

labels,imgs = removeMGnum(labels,images)

t_labels,t_imgs = removeMGnum(t_labels,t_images)

## Shrinking the data can make the simulation faster
#imgs = p_shrink(imgs)
#t_imgs = p_shrink(t_imgs)
print('Pretreatment finished!!')

## Print the 'k'th handwriting number in a simple way
### Test the consistency of label and imgs
'''
k=8
for i in range(28):
  for j in range(28):
    if imgs[k][i*28+j]==0:
      print('-',end=' ')
    else:
      print('0',end=' ')
      
  print('\n')
print('No.',labels[k])
'''

## sort function sort out image data to get what we want to classify
def sort(num,label,imgs):
  dum = []
  Ct = 0
  for i in label:
    if i==num:
      dum.append(imgs[Ct])
    Ct+=1
  return(dum)

#Here we choose to identify 5 and 9
imgs_0 = sort(5,labels,imgs)
imgs_1 = sort(9,labels,imgs)
t_imgs_0 = sort(5,t_labels,t_imgs)
t_imgs_1 = sort(9,t_labels,t_imgs)

print(np.shape(imgs_1))

## Here's the logisticregression class using gradient descent as optimizing method


class LogisticRegression:
  def __init__(self,x1,x2):
    self.n1,dum = np.shape(x1)
    self.n2,dum = np.shape(x2)

    y1 = np.zeros(self.n1)
    y2 = np.ones(self.n2)

    self.X = np.hstack((x1.T,x2.T)).T
    self.Y = np.hstack((y1,y2)).reshape(1,self.n1+self.n2)

  def sigmoid(self,x):
    return(1/(1+np.exp(-x)))
  
  def h(self,theta):
    pred = []
    for x in self.X:
      pred.append(self.sigmoid(theta.dot(x)[0]))
    return(np.array(pred))

  def loss(self,theta):

    alpha = 0.005

    hf = self.h(theta)

    error = hf-self.Y

    dJ = np.dot(error[0].T,self.X)+alpha*2*theta

    return(error,dJ)

  theta = np.zeros((1,7**2))

  def MMT(self,iters,step,theta,DC):
    error_list = []
    dJ = 0
    for iter in range(iters):
      if iter%10==0:
        print('No.',iter)
  
      error,dJ_n = self.loss(theta)
      theta+=-step*((1-DC)*dJ+(DC)*dJ_n)
      error_list.append(sum(error[0]**2))
    return(theta,error_list)



  def MMTRegression(self,iter,LR,theta,DC):
    
    theta_h,error = self.MMT(iter,LR,theta,DC)
    return(theta_h,error)


x1 = np.array(imgs_0)
x2 = np.array(imgs_1)
p = LogisticRegression(x1,x2)

theta = np.zeros((1,28**2))
#theta_h,error = p.GDRegression(100,0.00001,theta)##(iter,learning rate)
theta_h,error = p.MMTRegression(100,0.00001,theta,0.3)
print('finish!!!')


    


plt.plot(error)
plt.ylabel('Error')
plt.xlabel('Iteration')

## Test accuracy

t_x1 = np.array(t_imgs_0)
t_x2 = np.array(t_imgs_1)

##print(1/(1+np.exp(-theta@t_x2[600])))

## Total accuracy
def acc(m,i):
  return(1/(1+np.exp(-theta@m[i])))

n1,dum=np.shape(t_x1)
n2,dum=np.shape(t_x2)

error1 = 0
error2 = 0
for i in range(n1):
  
  error1+=acc(t_x1,i)
print('Accuracy of class1:',1-error1/n1)
for i in range(n2):
  
  error2+=acc(t_x2,i)
print('Accuracy fo class2:',error2/n2)