import numpy as np  
from multiprocessing  import Pool
# iSigma={}
import scipy

# def gauss(X,U,Sigma):
# 	det=np.linalg.det(Sigma)
# 	print 'fenmu'
# 	# print 'x-u',(X-U).shape,(X-U).dot(np.linalg.inv(Sigma)),'--------'
# 	print '----',np.sum(-0.5*((X-U).dot(np.linalg.inv(Sigma))*(X-U)),axis=1)
# 	fenmu= np.exp(np.sum(-0.5*((X-U).dot(np.linalg.inv(Sigma))*(X-U)),axis=1)-0.5*\
# 				np.trace(
# 					iSigma.dot(
# 						np.linalg.inv( Sigma) ),axis1=2,axis2=3))
# 	print 'fenmu',fenmu
# 	fenzi=(2*np.pi)**(0.5*X.shape[1])*det**(0.5)
# 	return fenmu/fenzi
def gauss(X,U,Sigma):
	
	r=scipy.stats.multivariate_normal.pdf(x=X,mean=U,cov=Sigma)
	
	return r
	
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e2, np.abs(x) + np.abs(y))))
class EM_extended:
	def __init__(self,U,Sigma,Pi,M=64,max_iteration=100,toi=1e-2,verbose=1):
		self.K,self.D,self.feature_size=U.shape
		self.M=M
		self.U=np.random.random([M,self.feature_size])*0.0001+np.mean(U,axis=0).mean(axis=0)### M*192
		self.Sigma=np.eye(self.feature_size)+np.zeros([M,self.feature_size,self.feature_size])#M*192*192
		# # print self.Sigma,np.linalg.inv(self.Sigma)
		# print self.Sigma.shape
		self.Pi=np.ones(M)/M #M
		self.iU=U#K*D*192
		self.iSigma=Sigma#K*D*192*192
		self.iPi=Pi#K*D
		self.max_iteration=max_iteration
		self.toi=toi
		self.verbose=verbose
		self.h=np.zeros([self.K,self.D,self.M])##K,D,M
		

	def train(self):
		M,K,D=self.M,self.K,self.D
		p=Pool(1)
		pre_param=(self.U,self.Sigma,self.Pi)

		ii=0
		error=self.toi+1
		while(ii<self.max_iteration and error>self.toi ):
			print 'error', error
			if( self.verbose>0):
				if ii%self.verbose==0:print "iteration %s" %ii
			ii+=1

			# E STEP
			gauss_p= lambda x:gauss(self.iU.reshape(K*D,self.feature_size),x[0],x[1])
			result=map(gauss_p,list(zip(self.U,self.Sigma))) #M,K,D
 			h_g=np.transpose(np.array(result).reshape(M,K,D),[1,2,0])\
				.reshape(K,D,M)#K,D,M
			h_function=lambda x:np.exp(-0.5*\
				np.trace(self.iSigma.dot(np.linalg.inv( x) ),axis1=2,axis2=3))
			result=map(h_function,self.Sigma) #M,K*D

			h_e=np.transpose(np.array(result),[1,2,0])
			fenmu=np.power(h_g*h_e,self.iPi.reshape(K,D,1))*(self.Pi)
			fm= (np.sum(fenmu,axis=2).reshape(K,D,1))
			eps=1e-6
			self.h=(fenmu+eps/M)/(fm+eps)
			print self.h
 
			# M STEP
			self.Pi=self.h.sum(axis=0).sum(axis=0)/(D*K)
			w=self.h*(self.iPi.reshape(K,D,1))
			w/=(w.sum(axis=0).sum(axis=0))
			self.U=(w.reshape(K,D,M,1)*(self.iU.reshape(K,D,1,self.feature_size)))\
				.sum(axis=0).sum(axis=0)
			iSigma_t=self.iSigma.reshape(K,D,1,self.feature_size,self.feature_size)
			sigma_iu=self.iU.reshape(K,D,1,self.feature_size,1)
			sigma_u=self.U.reshape(1,1,M,self.feature_size,1)
			u_delta=sigma_iu-sigma_u
			u_delta2=u_delta.reshape(K,D,M,1,self.feature_size)
			sigma_w=w.reshape(K,D,M,1,1)
			sigma_tmp=sigma_w*(iSigma_t+(u_delta)*(u_delta2))#K D M feature feature
			print sigma_tmp.shape
			self.Sigma=sigma_tmp.sum(axis=0).sum(axis=0)

			error1=rel_error(self.U,pre_param[0])
			error2=rel_error(self.Sigma,pre_param[1])
			# np.max(np.abs(self.U-pre_param[0]))/np.max(np.abs(self.U))
			# error2=np.max(np.abs(self.Sigma-pre_param[1]))/np.max(np.abs(self.Sigma))
			# # error3=np.max(np.abs(self.Pi-pre_param[2]))
			error=max(error1,error2)
			 
		p.terminate()
		if ii==self.max_iteration:
			print 'warning: not converge'
		else:
			print 'converge'
		return self
	def score(self,X):
		for ii in range(self.M):
		results=map(lambda u,cov,pi:pi*gauss(X,u,cov),zip(self.U,self.Sigma,self.U))
		return np.sum(results)
# def gauss(X,U,Sigma):
# 	det=np.linalg.det(Sigma)
# 	return (0.5*det**0.5)*np.exp(-0.5*((X-U).T.dot(np.linalg.inv(Sigma)).dot(X-U)))



# def gauss(X,U,Sigma):
# 	det=np.abs(np.linalg.det(Sigma))
# 	print 'fenmu'
# 	# print 'x-u',(X-U).shape,(X-U).dot(np.linalg.inv(Sigma)),'--------'
# 	# print '----',np.sum(-0.5*((X-U).dot(np.linalg.inv(Sigma))*(X-U)),axis=1).shape
# 	# print np.sum(-0.5*((X-U).dot(np.linalg.inv(Sigma))*(X-U)),axis=1)-0.5*\
# 	# 			np.trace(
# 	# 				iSigma[0].dot(
# 	# 					np.linalg.inv( Sigma) ),axis1=2,axis2=3).reshape(X.shape[0])
# 	# X,U,Sigma=X.astype(np.float128),U.astype(np.float128),Sigma.astype(np.float128)
# 	fenmu= np.exp(np.sum(-0.5*((X-U).dot(np.linalg.inv(Sigma))*(X-U)),axis=1))-0.5*\
# 				np.trace(
# 					iSigma[0].dot(
# 						np.linalg.inv( Sigma) ),axis1=2,axis2=3).reshape(X.shape[0]))
# 	# print 'fenmu',fenmu
# 	print 'finish'
# 	print X.shape[1]
# 	print 1,det
# 	det_=det**(0.5)
# 	print 2
# 	temp=(2*np.pi)**(0.5*X.shape[1])
# 	print 3
# 	fenzi=temp*det_,
# 	print temp,  det_
# 	print 'finally',fenzi
# 	return fenmu/fenzi


def load_data():
	all_data=np.load('dct_feature.npz')
	# data=all_data['data']
	# labels=all_data['labels']
	# label_str=all_data['label_str']
	return all_data
class Solver:
	def __init__(self,class_label):
		weights,means,cov=[],[],[]
		class_labels_images=np.arange(5000)[load_data()['labels'][class_label]>1]
		for ii in class_labels_images:
		    model=joblib.load('/home/cy/tmp_model/model%s' %ii)
		    weights.append(model.weights_)
		    means.append(model.means_)
		    cov.append(model.covariances_)
		Pi=np.array(weights)
		U=np.array(means)
		Sigma=np.array(cov)
		U=np.transpose(U,[1,0,2])
		Sigma=np.transpose(Sigma,[1,0,2,3])
		m=EM_extended(U,Sigma,Pi)
		self.m=m.train()
		
	
	def test(self,image):
		result =0
		for ybr in image:
			result+=(np.log(self.m.score(ybr)))
		return result
	def test_all(self):
		'''

		'''
		data=load_data()['data']
		N,H,W,Chanel,windows1,windows2=data.shape
		images=data.reshape(N,H*W,Chanel*windows1*windows2)
		p=Pool(16)
		scores=p.map(lambda x:self.test(x),images)
		p.terminate()
		return scores
