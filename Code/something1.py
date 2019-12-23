import numpy as np

def vect_sub(x1,x2,n):
	subs=[]
	for i in range(n):
		subs.append(x1[i]-x2[i])
	return subs

def transpose(matrix,m,n):
	newmat=[]
	for i in range(n):
		newmat.append([])
	for i in range(n):
		for j in range(m):
			newmat[i].append(matrix[j][i])
	return newmat

def tree_dist(x1,x2,syntax_tree,sentence_length):
	subvector=[]
	dist=0
	for i in range(sentence_length):
		subvector.append(syntax_tree[x1][i]-syntax_tree[x2][i])
	for i in subvector:
		if(i):
			dist=dist+1
	return dist

def transform_dist(x1,x2,B_mat,m,n):
	subt=[]
	subt.append(vect_sub(x1,x2,n))
	subtT=transpose(subt,1,n)
	dist=np.dot(transpose(np.dot(B_mat,subtT),m,1),np.dot(B_mat,subtT))
	return dist

def gradient(a,b,B_mat,h):
	summation=0
	for i in range(b):
		summation+=abs(B_mat[a][i]*h[i])
	return 2*summation*h[b]

def gradient_descent(word_vectors,syntax_tree,B_mat,alpha,iterations,m,n):
	sl=len(word_vectors)
	while(True):
		e=0
		for i in range(sl):
			for j in range(sl):
				e+=abs(tree_dist(i,j,syntax_tree,sentence_length)-transform_dist(word_vectors[i],word_vectors[j],B_mat,m,n)**2)
		e=e/sl**2
		for i in range(m):
			for j in range(n):
				B_mat[i][j]=B_mat[i][j] - alpha*(gradient(i,j,B_mat,vect_sub(word_vectors[i],word_vectors[j],n))/sl**2)

		print(e)
		print(B_mat)
		print()
		print()
	pass

sentence="hi my name is good name"
sentence_length=len(list(sentence.split(" ")))
print(sentence_length)
dimension=6
word_vectors=[]
for i in range(sentence_length):
	word_vectors.append(np.random.normal(0,0.1,dimension))
#print(word_vectors)
syntax_tree=[[0,0,0,0,0,0],[1,0,0,0,0,0],[0,1,0,0,0,0],[1,0,1,0,0,0],[1,0,0,1,0,0],[1,0,0,1,1,0]]
b=3
#B_mat=np.random.rand(b,dimension)
B_mat=[]
for i in range(b):
	B_mat.append([])
for i in range(b):
	for j in range(dimension):
		B_mat[i].append(np.random.rand())

print(B_mat)
print(np.dot(transpose(B_mat,b,dimension),B_mat))
something=transform_dist(word_vectors[1],word_vectors[2],B_mat,b,dimension)
gradient_descent(word_vectors,syntax_tree,B_mat,0.01,100,b,dimension)
print(transform_dist(word_vectors[1],word_vectors[2],B_mat,b,dimension))
print(tree_dist(1,2,syntax_tree,sentence_length))
print(something)