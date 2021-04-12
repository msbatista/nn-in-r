# weight.i <-
#   0.01 * matrix(
#     data = rnorm(layer.size(i) * layer.size(i + 1), sd = 0.5),
#     nrow = layer.size(i),
#     ncol = layer.size(i + 1)
#   )
# bias.i <- matrix(0, nrow = 1, ncol = layer.size(1))

# input <- matrix(1:4, nrow = 2, ncol = 2)
# weights <- matrix(1:6, nrow = 2, ncol = 3)
# bias <- matrix(1:3, nrow = 1, ncol = 3)
#
# s1 <- input %*% weights + matrix(rep(bias, each = 2), ncol = 3)
#
# s2 <- sweep(input %*% weights, 2, bias, '+')
# all.equal(s1, s2)
# max(0, s1)

predict.dnn <- function(model, data) {
  new.data <- data.matrix(data)
  hidden.layer <- sweep(new.data %*% model$W1, 2, model$b1, '+')

  hidden.layer <- pmax(hidden.layer, 0)
  score <- sweep(hidden.layer %*% model$W2, 2, model$b2, '+')

  score.exp <- exp(score)
  probs <- sweep(score.exp, 1, rowSums(score.exp), '/')

  labels.predicted <- max.col(probs)
  return(labels.predicted)
}

train.dnn <- function(x,
                      y,
                      train_data = data,
                      test_data = NULL,
                      hidden = c(6),
                      max_it = 2000,
                      abstol = 1e-2,
                      lr = 1e-2,
                      reg = 1e-3,
                      display = 100,
                      random.seed = 1) {
  set.seed(random.seed)

  N <- nrow(train_data)

  X <- unname(data.matrix(train_data[, x]))
  Y <- train_data[, y]

  if (is.factor(Y)) {
    Y <- as.integer(Y)
  }

  Y.index <- cbind(1:N, Y)

  D <- ncol(X)
  K <- length(unique(Y))
  H <- hidden

  W1 <- 0.01 * matrix(rnorm(H * K, sd = 0.5), nrow = D, ncol = H)
  b1 <- matrix(0, nrow = 1, ncol = H)

  W2 <- 0.01 * matrix(rnorm(H * K, sd = 0.5), nrow = H, ncol = K)
  b2 <- matrix(0, nrow = 1, ncol = K)

  batch_size <- N

  i <- 0
  loss <- 0

  while (loss < abstol) {
    i <- i + 1

    if (i == max_it) {
      cat("Max iterations reached...\nStoping\n")
      break
    }

    hidden.layer <- sweep(X %*% W1, 2, b1, '+')

    # ReLu
    hidden.layer <- pmax(hidden.layer, 0)
    score <- sweep(hidden.layer %*% W2, 2, b2, '+')

    # SoftMax
    score.exp <- exp(score)
    probs <- sweep(score.exp, 1, rowSums(score.exp), '/')

    corect.logprobs <- log(probs[Y.index])
    data.loss <- sum(corect.logprobs) / batch_size
    reg.loss <- 0.5 * reg * (sum(W1 * W1) + sum(W2 * W2))
    loss <- data.loss + reg.loss

    if (i %% display == 0) {
      if (!is.null(test_data)) {
        model <- list(
          D = D,
          H = H,
          K = K,
          W1 = W1,
          b1 = b1,
          W2 = W2,
          b2 = b2
        )

        labs <- predict.dnn(model, test_data[, -y])
        accuracy <- mean(as.integer(test_data[, y]) == labs)
        message(sprintf(
          "Iteration: %d -- Loss: %g -- Accuracy: %g\n",
          i,
          loss,
          accuracy
        ))
      } else {
        message(sprintf("Iteration: %d -- Loss: %g\n", i, loss))
      }
    }

    dscores <- probs
    dscores[Y.index] <- dscores[Y.index] - 1
    dscores <- dscores / batch_size

    dW2 <- t(hidden.layer) %*% dscores
    db2 <- colSums(dscores)

    dhidden <- dscores %*% t(W2)
    dhidden[hidden.layer <= 0] <- 0

    dW1 <- t(X) %*% dhidden
    db1 <- colSums(dhidden)

    dW2 <- dW2 + reg * W2
    dW1 <- dW1 + reg * W1

    W1 <- W1 - lr * dW1
    b1 <- b1 - lr * db1

    W2 <- W2 - lr * dW2
    b2 <- b2 - lr * db2
  }

  model <- list(
    D = D,
    H = H,
    K = K,
    W1 = W1,
    b1 = b1,
    W2 = W2,
    b2 = b2
  )

  return(model)
}
###############################################################################
# Testing
###############################################################################
set.seed(1)

summary(iris)
plot(iris)

samp <- c(sample(1:150, 25),
          sample(51:100, 25),
          sample(101:150, 25))

ir.model <- train.dnn(
  x = 1:4,
  y = 5,
  train_data = iris[samp,],
  test_data = iris[-samp,]
)

labels.dnn <- predict.dnn(ir.model, iris[-samp, -5])

table(iris[-samp, 5], labels.dnn)

mean(as.integer(iris[-samp, 5]) == labels.dnn)

###############################################################################
library(nnet)

ird <- data.frame(rbind(iris3[, , 1], iris3[, , 2], iris3[, , 3]),
                  species = factor(c(rep("s", 50), rep("c", 50), rep("v", 50))))
ir.nn2 <-
  nnet(
    species ~ .,
    data = ird,
    subset = samp,
    size = 6,
    rang = 0.1,
    decay = 1e-2,
    maxit = 2000
  )

labels.nnet <- predict(ir.nn2, ird[-samp, ], type = "class")
table(ird$species[-samp], labels.nnet)

mean(ird$species[-samp] == labels.nnet)
