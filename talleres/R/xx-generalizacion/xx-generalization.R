###
### Celda 1
###


set.seed(12346)

N = 20            # tama??o de la muestra
x = runif(N)      # ejemplos
noise = 0.05 * rnorm(N)
df_sample = data.frame(x = x, 
                       y = noise + 0.5 + 0.4 * sin(2 * pi * x))

####
#### Celda 2
#### 

##
## Conjunto de prueba
##
index = sample(1:20, 10)
df_train = df_sample[index,]
df_test  = df_sample[-index,]

##
## real
##
x_real = seq(from = 0, to = 1, length.out = 100)
y_real = 0.5 + 0.4 * sin(2 * pi * x_real)


####
#### Celda 3
####
P = 5

m = lm(y ~ poly(x, P), data = df_train)

y_train_pred = predict(m, df_train)
y_test_pred  = predict(m, df_test)

y_model = predict(m, data.frame(x=x_real))

plot(x_real, y_real, type='l', ylim=c(0,1))
points(df_train$x, df_train$y, lwd = 3)
points(df_train$x, y_train_pred, col='red', lwd = 3)

points(df_test$x, df_test$y, col='blue', lwd = 3)
#points(df_test$x, y_test_pred, col='blue', lwd = 3)

lines(x_real, y_model, type = 'l', lty = 3, col='red')
grid()