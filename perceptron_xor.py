import numpy as np
entrada = []
entrada.append(np.array([1, 1]))
entrada.append(np.array([1, 0]))
entrada.append(np.array([0, 1]))
entrada.append(np.array([0, 0]))
print(entrada)



def andgate(entrada):
    saida = np.array([1, 0, 0, 0])
    pesos = np.zeros(no_entradas + 1)
    train(pesos,saida)
    
    return predict(entrada,pesos)

def orgate(entrada):
    saida = np.array([1, 1, 1, 0])
    pesos = np.zeros(no_entradas + 1)
    train(pesos,saida)
    return predict(entrada,pesos)

def nandgate(entrada):
    saida = np.array([0, 1, 1, 1])
    pesos = np.zeros(no_entradas + 1)
    train(pesos,saida)
    
    return predict(entrada,pesos)

def xorgate(entrada):
    
    y2=nandgate(entrada)
    y3=orgate(entrada)
    y1 = andgate([y2,y3])
    
    return y1


epocas=1000
learning_rate=0.01
no_entradas = len(entrada[0])
#definindo os pesos



def ativacao(soma):
    if soma > 0:
        activacao = 1
    else:
        activacao = 0            
    return activacao


def predict(e,p):
    #np.dot produto dos vetores a x b
    soma=0
    #print(p[:no_entradas])
    soma = np.dot(e,p[:no_entradas])
    #ao final soma o bias    
    soma = soma + p[no_entradas]
    
    return ativacao(soma)

def train(pesos,saida):
    for _ in range(epocas):
        xy =zip(entrada, saida) 
        for inputs, label in xy:
            prediction = predict(inputs,pesos)
            pesos[0] += learning_rate * (label - prediction) * inputs[0]
            pesos[1] += learning_rate * (label - prediction) * inputs[1]
            pesos[2] += learning_rate * (label - prediction)


ys = []
for e in entrada:
    ys.append(xorgate(e))

print(ys)   