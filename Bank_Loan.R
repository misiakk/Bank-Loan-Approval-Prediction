install.packages("Amelia")
install.packages('rpart')
install.packages('rpart.plot')

library(Amelia)
library(dplyr)
library(caTools)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(tidyverse)

# Zbiór zawierający dane klientów banku, którym przyznano lub nie przyznano pożyczkę
train <- read.csv('C:\\Users\\user\\PPwMI\\Loan_Train.csv')

missmap(train, main="Credit Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

# 'Y' oznacza pozytywny status kredytu (1) i 'N' oznacza negatywny status kredytu (0)
train <- train %>%
  mutate(Loan_Status = ifelse(Loan_Status == 'Y', 1, 0))

# Wstawienie "Other" w puste miejsca w kolumnie "Gender"
train <- train %>%
  mutate(Gender = ifelse(Gender == "" | is.na(Gender), "Other", Gender))

# Jeżeli puste -> uznajemy ze brak historii kredytowej
train <- train %>%
  mutate(Credit_History = ifelse(is.na(Credit_History), 0 , Credit_History))

#Jeżeli puste to uznajemy, że klient nie jest samozatrudniony
train <- train %>%
  mutate(Self_Employed = ifelse(Self_Employed == "" | is.na(Self_Employed), "No" , Self_Employed))

# Jeżeli puste, to przyjmujemy że klient nie jest married
train <- train %>%
  mutate(Married = ifelse(Married == "" | is.na(Married), "No" , Married))

#Jeżeli puste to uznajemy że klient nie ma osób na utrzymaniu
train <- train %>%
  mutate(Dependents = ifelse(Dependents == "" | is.na(Dependents), "0" , Dependents))

ggplot(train,aes(Loan_Status)) + geom_bar()
ggplot(train,aes(Education)) + geom_bar(aes(fill=factor(Education)),alpha=0.5)
ggplot(train,aes(Property_Area)) + geom_bar(aes(fill=factor(Property_Area)),alpha=0.5)
ggplot(train,aes(Gender)) + geom_bar(aes(fill=factor(Gender)),alpha=0.5)
ggplot(train,aes(Credit_History)) + geom_bar(aes(fill=factor(Credit_History)),alpha=0.5)
ggplot(train,aes(Education)) + geom_bar(aes(fill=factor(Education)),alpha=0.5)
ggplot(train,aes(Self_Employed)) + geom_bar(aes(fill=factor(Self_Employed)),alpha=0.5)
ggplot(train,aes(Married)) + geom_bar(aes(fill=factor(Married)),alpha=0.5)
ggplot(train,aes(Dependents)) + geom_bar(aes(fill=factor(Dependents)),alpha=0.5)
ggplot(train,aes(ApplicantIncome)) + geom_histogram(fill='blue',bins=20,alpha=0.5)
ggplot(train,aes(CoapplicantIncome)) + geom_histogram(fill='blue',bins=20,alpha=0.5)
ggplot(train,aes(LoanAmount)) + geom_histogram(fill='blue',bins=20,alpha=0.5)

#Wykres pokazujący rozkład wielkości pożyczki w podziale na ilość osób, które ma na utrzymaniu
pl <- ggplot(train,aes(Dependents,LoanAmount)) + geom_boxplot(aes(group=Dependents,fill=factor(Dependents),alpha=0.4)) 
pl + scale_y_continuous(breaks = seq(min(0), max(700), by = 15))

# Funkcja, która mapuje braki danych w kolumnie LoanAmount według median obliczonych per liczba osób na utrzymaniu
impute_loan <- function(loan,dependent){
  out <- loan
  for (i in 1:length(loan)){
    
    if (is.na(loan[i])){
      
      if (dependent[i] == "0"){
        out[i] <- 120
        
      }else if (dependent[i] == "1"){
        out[i] <- 137
       }else if (dependent[i] == "2"){
          out[i] <- 135
      }else{
        out[i] <- 134
      }
    }else{
      out[i]<-loan[i]
    }
  }
  return(out)
}

fixed.loan <- impute_loan(train$LoanAmount,train$Dependents)

train$LoanAmount <- fixed.loan

missmap(train, main="Credit Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

# Wykres pokazujący rozkład okresu kredytowania w podziale na ilość osób, które ma na utrzymaniu
pl1 <- ggplot(train,aes(Dependents,Loan_Amount_Term)) + geom_boxplot(aes(group=Dependents,fill=factor(Dependents),alpha=0.4)) 
pl1 + scale_y_continuous(breaks = seq(min(0), max(500), by = 10))
# Wstawienie wartości 360 w puste miejsca w kolumnie Loan_Amount_Term
train <- train %>%
  mutate(Loan_Amount_Term = ifelse(is.na(Loan_Amount_Term), 360 , Loan_Amount_Term))

str(train)

#Usunięcie kolumny Loan_id ze zbioru danych
train <- select(train,-Loan_ID)

# Model logitowy
log.model <- glm(formula=Loan_Status ~ . , family = binomial(link='logit'),data = train)
summary(log.model)

set.seed(101)

split = sample.split(train$Loan_Status, SplitRatio = 0.70)

final.train = subset(train, split == TRUE)
final.test = subset(train, split == FALSE)

final.log.model <- glm(formula=Loan_Status ~ . , family = binomial(link='logit'),data = final.train)
summary(final.log.model)

fitted.probabilities <- predict(final.log.model,newdata=final.test,type='response')

fitted.results <- ifelse(fitted.probabilities > 0.5,1,0)

misClasificError <- mean(fitted.results != final.test$Loan_Status)

# Dokładność modelu 
print(paste('Accuracy',1-misClasificError))

# Stworzenie tabeli kontyngencji, któr aporównuje rzeczywiste wartości przyznania kredytu z wynikami klasyfikacji
table(final.test$Loan_Status, fitted.probabilities > 0.5)


# Decision trees
str(train)

tree <- rpart(Loan_Status ~ . , method='class', data= train)

plot(tree, uniform=TRUE, main="Drzewo decyzyjne przyznania pożyczki")
text(tree, use.n=TRUE, all=TRUE)

prp(tree)

### NEURAL NETWORKS ###

install.packages(c('neuralnet','keras','tensorflow'),dependencies = T)
library(tidyverse)
library(neuralnet)

colnames(train)
train_1<- train[, c("Loan_Status", "CoapplicantIncome", "Credit_History", "Property_Area", "Married", "ApplicantIncome")]

train_1 <- train %>% mutate_if(is.character, as.factor)
str(train_1)


set.seed(245)
data_rows <- floor(0.80 * nrow(train_1))
train_indices <- sample(c(1:nrow(train_1)), data_rows)
train_data <- train_1[train_indices,]
test_data <- train_1[-train_indices,]


model = neuralnet(
  Loan_Status~CoapplicantIncome+Credit_History+ApplicantIncome,
  data=train_data,
  hidden=c(4,2),
  linear.output = FALSE
)

plot(model)

pred <- predict(model, test_data)
#labels <- c("Dostał pożyczkę", "Nie dostał pożyczki")
prediction_label <- data.frame(max.col(pred)) %>%     
 # mutate(pred=labels[max.col.pred.]) %>%
  #select(2) %>%
  unlist()

table(test_data$Loan_Status, prediction_label)

check = as.numeric(test_data$Loan_Status) == max.col(pred)
accuracy = (sum(check)/nrow(test_data))*100
print(accuracy)

