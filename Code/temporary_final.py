import numpy as np
from bert_embedding import BertEmbedding
import nltk

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
	for i in range(sentence_length-1):
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

def gradient(a,b,B_mat,m,n,sl,word_vectors):
	summations=[]
	for i in range(sl):
		for j in range(sl):
			summation=0
			for x in range(n):
				summation+=abs(B_mat[a][x]*(word_vectors[i][x]-word_vectors[j][x]))
			summations.append(2*summation*(word_vectors[i][b]-word_vectors[j][b]))
	return sum(summations)	

def gradient_descent(word_vectors,syntax_tree,B_mat,alpha,iterations,m,n):
	sl=len(word_vectors)
	#print("sl: %d"%sl)
	while(True):
		e=0
		for i in range(sl):
			for j in range(sl):
				e+=abs(tree_dist(i,j,syntax_tree,sentence_length)-transform_dist(word_vectors[i],word_vectors[j],B_mat,m,n))
		e=e/sl**2
		print("%f"%e)
		#print(B_mat)    		
		for i in range(m):
			for j in range(n):
				#print("before: %.15f"%B_mat[i][j])
				B_mat[i][j]=B_mat[i][j] - alpha*(gradient(i,j,B_mat,m,n,sentence_length,word_vectors))
				#print("after: %.15f"%B_mat[i][j])
		#print(e)
		#print(B_mat)
		print()
		print()
	pass

sentence="hi my name is good name"
sentence_length=len(list(sentence.split(" ")))
embedding_bert=BertEmbedding()
embeddings=embedding_bert(sentence.split(" "))
word_vectors=[]
#print("HELLO:\n")
#print(embeddings)
dimension=len(embeddings[1][1][0])
#print(dimension)
for i in range(sentence_length):
	word_vectors.append(embeddings[i][1][0])
#print(word_vectors[0][8])
#tree=nltk.CFG.fromstring(sentence)
#print(tree)
syntax_tree=[[0,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,1,1]]
b=3
#B_mat=np.random.rand(b,dimension)
B_mat=[]
for i in range(b):
	B_mat.append([])
for i in range(b):
	for j in range(dimension):
		B_mat[i].append(np.random.random_integers(0,10))
#print(B_mat)
#print(np.dot(transpose(B_mat,b,dimension),B_mat))
#something=transform_dist(word_vectors[1],word_vectors[2],B_mat,b,dimension)
gradient_descent(word_vectors,syntax_tree,B_mat,0.3,100,b,dimension)
#print(transform_dist(word_vectors[1],word_vectors[2],B_mat,b,dimension))
#print(tree_dist(1,2,syntax_tree,sentence_length))
#print(something)
