# Chercher le meilleur modèle de régression pour prédire le taux de
# crimes violents aux Etats-Unis (ViolentCrimesPerPop) du dataset
# les modèles seront comparés sur la base RMSE évaluée par 10-fold CV

setwd("/home/nathan/fac/m1/s2/apprentissage_supervise")
file <- 'violent_crimes.txt'
violent_crimes <- read.csv(file, header=FALSE)
violent_crimes[violent_crimes == "?"] <- NA # remplace "?" par NA

# ==== Uniformisation des données ==== 

# variables V1 ... V5 inutiles
modified_crimes <- violent_crimes[,6:147]


# supprimer toutes les lignes qui contiennent au moins un Na 
modified_crimes <- modified_crimes[complete.cases(modified_crimes), ]


for (i in colnames(modified_crimes)){ 
  # On transforme les données
  modified_crimes[[i]] <- as.numeric(modified_crimes[[i]]) # OK
  # on remplace par la moyenne de la colonne, en omettant les valeurs Na de la colonne ?
  #modified_crimes[[i]][is.na(modified_crimes[[i]])] <- mean(modified_crimes[[i]], na.rm = T)
}

# ==== Initialisation de variables ====

X = as.matrix(modified_crimes[ , -141]) # tout sauf -141
Y = as.matrix(modified_crimes[ , 141]) # seulement la variable à prédire

variables <- paste("V", 6:145, sep = "")
variables <- append(variables,"V147") # On rajoute la variable 'non-violent crimes'
formula <- reformulate(variables, response = "V146") # équivaut à : V146 ~ V3 + V4 + ... + V145 + V147


#  =====  Régression linéaire avec forward  ===== OK

min.model <- lm(V146 ~ 1,data = modified_crimes)
tmp = "~"
for(n in seq (6,145,1)) tmp=paste(tmp,' + V',n,sep='')
tmp = paste(tmp,' + V147',sep='') # on ajoute aussi les crimes non-violents
# Sélection de 50 variables
model1b <- step(min.model, direction = "forward", scope = (tmp),steps = 50,trace = F) 
summary(model1b)
# => Call :  lm(formula = V146 ~ V137 + V135 + V133 + V131 + V100 + V54 + 
# V129 + V77 + V8 + V128 + V114 + V122 + V96 + V143 + V76 + 
# V121, data = modified_crimes)


# ==== Régression Backward ==== OK

fitAll <- lm(V146 ~ .,data = modified_crimes)
modelBackward <- step(fitAll,direction="backward")
summary(modelBackward)

#Call : lm(formula = V146 ~ V16 + V19 + V21 + V22 + V26 + V27 + V33 + 
#  V39 + V42 + V43 + V48 + V49 + V51 + V54 + V61 + V63 + V65 + 
#  V67 + V71 + V72 + V74 + V84 + V86 + V87 + V91 + V93 + V94 + 
#  V96 + V97 + V99 + V101 + V105 + V107 + V113 + V116 + V119 + 
#  V120 + V121 + V123 + V128 + V131 + V132 + V133 + V135 + V136 + 
#  V137 + V138 + V139 + V140 + V141 + V143 + V145 + V147, data = modified_crimes)

# ==== Stepwise ==== (Même résultat que Backward)

# le modèle à conserver est le dernier obtenu 
full.model <- lm(V146 ~ .,data = modified_crimes) # V146 ~ . = everything except V146
library(MASS)
step.model <- stepAIC(full.model, direction = "both", trace = TRUE)
summary(step.model)

#Call : lm(formula = V146 ~ V16 + V19 + V21 + V22 + V26 + V27 + V33 + 
#V39 + V42 + V43 + V48 + V49 + V51 + V54 + V61 + V63 + V65 + 
#V67 + V71 + V72 + V74 + V84 + V86 + V87 + V91 + V93 + V94 + 
#V96 + V97 + V99 + V101 + V105 + V107 + V113 + V116 + V119 + 
#V120 + V121 + V123 + V128 + V131 + V132 + V133 + V135 + V136 + 
#V137 + V138 + V139 + V140 + V141 + V143 + V145 + V147, data = modified_crimes)

# ==== Régression PCR ==== OK
library(pls)

model2 <- pcr(formula, data = modified_crimes, validation = "LOO")

summary(model2) # Number of components considered: 141
plot(RMSEP(model2), legendpos = "topright")

# 57 composantes semblent le plus efficace. 
# On peut tracer les coefficients obtenus avec 57 composantes
plot(model2, plottype = "coef", ncomp = 57, legendpos = "topleft",xaxt='n')
axis(1,at = c(1:140,142),labels = colnames(modified_crimes)[-141],las = 2) # exclut la colonne 141 (à prédire)
plot(model2, ncomp = 57, asp = 1, line = TRUE,col="blue") # Validation


# ==== Régression PLS ===== OK
library(plsdepot)
tmp = plsreg1(X,modified_crimes[,141,drop=F],comps=10,crosval=T)
print(tmp$Q2)

model3 = plsreg1(X,modified_crimes[,141,drop=F],comps=57,crosval=T)
print(model3$Q2)
plot(model3,what='variables',comps=c(1,2))

plot(model3,what='observations',comps=c(1,2),show.names=T)

plot(modified_crimes$V146,model3$y.pred,type='n',xlab='Original',ylab='Predicted')
title('Qualité de prédiction', cex.main = 0.9)
abline(a = 0, b = 1, col = 'gray85', lwd = 2)
points(modified_crimes$V146, model3$y.pred, col = '#5592e3')

# ==== RIDGE ==== OK
library(MASS)

model_ridge <- lm.ridge(formula,data = modified_crimes,lambda=seq(0,20,0.1))
plot(seq(0,20,0.1),model_ridge$GCV,xlab='lambda',ylab='GCV',main='GCV')

print(model_ridge$lambda[which.min(model_ridge$GCV)]) # = 0 ou 9.4 avec toutes les données
best_lambda_ridge = model_ridge$lambda[which.min(model_ridge$GCV)]

#print(min(model_ridge$GCV)) 

# D’après l’indice de validation croisée généralisée (GCV), le labmbda optimal est 1. Ré-éstimons alors le modèle
model_ridge <- lm.ridge(formula,data = modified_crimes,lambda=best_lambda_ridge)
print(model_ridge$coef)

# ==== Lasso ==== OK
library(lars)

model_lasso <- lars(X,Y,type="lar",trace=F,normalize=TRUE)

plot(model_lasso,xvar='df', plottype='coeff')

cv = cv.lars(X,Y, K = 20, trace = FALSE, plot.it = TRUE, se = TRUE,type = "lar",mode='step')
print(model_lasso$lambda[which.min(cv$cv)]) # = meilleur lambda : 0.01262425
best_lambda_lasso = model_lasso$lambda[which.min(cv$cv)] 

# ==== Elastic Net ====
library(glmnet)
# Note that setting alpha equal to 0 is equivalent to using ridge regression 
# and setting alpha to some value between 0 and 1 is equivalent to using an elastic net. 

cv_model <- cv.glmnet(X, Y, alpha = 0.5)
best_lambda_EN <- cv_model$lambda.min
best_model <- glmnet(X, Y, alpha = 0.5, lambda = best_lambda_EN) # réestime le modèle avec le meilleur lambda

# ==== Comparaison des méthodes ====

library('pls')
library(glmnet)
library(randomForest)


# génère un échantillon aléatoire de taille nrow(modified_crimes)
# des nombres entiers de 1 à 10 avec remplacement
fold = sample(1:10,nrow(modified_crimes),replace=T)
pred = matrix(0,nrow(modified_crimes),8)

for (i in 1:10){
  # forward
  model1b = lm(V146 ~ V137 + V135 + V133 + V131 + V100 + V54 + 
                 V129 + V77 + V8 + V128 + V114 + V122 + V96 + V143 + V76 + 
                 V121,data=modified_crimes[-which(fold==i),])
  tmp = predict(model1b,modified_crimes[which(fold==i),])
  pred[which(fold==i),1] = tmp
  
  # backward (possible si p < n)
  modelBackward = lm(formula = V146 ~ V16 + V19 + V21 + V22 + V26 + V27 + V33 + 
                       V39 + V42 + V43 + V48 + V49 + V51 + V54 + V61 + V63 + V65 + 
                       V67 + V71 + V72 + V74 + V84 + V86 + V87 + V91 + V93 + V94 + 
                       V96 + V97 + V99 + V101 + V105 + V107 + V113 + V116 + V119 + 
                       V120 + V121 + V123 + V128 + V131 + V132 + V133 + V135 + V136 + 
                       V137 + V138 + V139 + V140 + V141 + V143 + V145 + V147, data=modified_crimes[-which(fold==i),])
  tmp = predict(modelBackward,modified_crimes[which(fold==i),])
  pred[which(fold==i),2] = tmp
  
  # pcr 
  modelPCR <- pcr(formula, data = data.frame(modified_crimes[-which(fold==i),]), validation = "LOO")
  tmp = predict(modelPCR,data.frame(modified_crimes[which(fold==i),]),ncomp=57) # on laisse 57, ça reste pertinent
  pred[which(fold==i),3] = tmp
  
  # pls
  model_pls = plsr(formula,data=data.frame(modified_crimes[-which(fold==i),]),scale=TRUE,ncomp=57)
  tmp = predict(model_pls,newdata=data.frame(modified_crimes[which(fold==i),]))
  pred[which(fold==i),4] = tmp[,,57]
  
  # ridge
  model_ridge <- lm.ridge(formula,data=data.frame(modified_crimes[-which(fold==i),]),lambda=best_lambda_ridge)
  tmp = scale(X[which(fold==i),], center = model_ridge$xm, scale = model_ridge$scales) %*% model_ridge$coef + model_ridge$ym
  pred[which(fold==i),5] = tmp
  
  # lasso
  model_lasso = lars(X[-which(fold==i),],Y[-which(fold==i)],type="lar",trace=F,normalize=TRUE)
  tmp = predict(model_lasso,X[which(fold==i),],s = best_lambda_lasso,mode="lambda")
  pred[which(fold==i),6] = tmp$fit
  
  # Elastic Net
  modelEN <- glmnet(X[-which(fold==i),],Y[-which(fold==i)], alpha = 0.5, lambda = best_lambda_EN)
  tmp = predict(modelEN, s = best_lambda_EN, newx = as.matrix(X[which(fold==i),]))
  pred[which(fold==i),7] = tmp
  
  # random forest
  modelRF <- randomForest(formula, data = data.frame(modified_crimes[-which(fold==i),]))
  tmp = predict(modelRF, newdata = data.frame(modified_crimes[which(fold==i),]))
  pred[which(fold==i),8] = tmp
  
  
  print("1 itération terminé")
}


listeCol = c("#66FF00","#3399FF","#FFFF00","#9933FF","#FF9933","#FF9999","#3300CC","#00CC99")
plot(modified_crimes$V146,pred[,1],col=listeCol[1],xlab="valeur réelle",ylab="valeur prédite")
points(modified_crimes$V146,pred[,2],col=listeCol[2])
points(modified_crimes$V146,pred[,3],col=listeCol[3])
points(modified_crimes$V146,pred[,4],col=listeCol[4])
points(modified_crimes$V146,pred[,5],col=listeCol[5])
points(modified_crimes$V146,pred[,6],col=listeCol[6])
points(modified_crimes$V146,pred[,7],col=listeCol[7])
points(modified_crimes$V146,pred[,8],col=listeCol[8])
abline(coef = c(0,1))
legend("topleft",legend=c('forward','backward','pcr','pls','ridge','lasso','ElasticNet','RandomForest'),col=listeCol,pch=1)

# ==== Erreur quadratique moyenne ====

colnames(pred)=c('forward','backward','pcr','pls','ridge','lasso','ElasticNet','RandomForest')
sorted_result <- sort(colMeans((pred-modified_crimes$V146)^2))

print(sorted_result) # Backward, Forward et Lasso => très performants
