#Install and Load Packages
install.packages("RODBC")
install.packages("xlsx")
install.packages('rJava')
library(RODBC)
library(xlsx)
library(rJava)

#Connect to Database
mycon<-odbcConnect("mysql_data",uid="root",pwd="********")
sqlTables(mycon)

#Load Data Set
historic_races<-sqlFetch(mycon,"historic_races",rownames="state")
historic_runners<-sqlFetch(mycon,"historic_runners",rownames="state")
daily_races<-sqlFetch(mycon,"daily_races",rownames="state")
daily_runners<-sqlFetch(mycon,"daily_runners",rownames="state")

#Intergate the four data sets
races <- merge(historic_races,daily_races,by = 'race_id')
runners <- merge(historic_runners,daily_runners,by = c('race_id','runner_id'))
df <- merge(races,runners,by='race_id')

#Delete the repeating variables
df_del_rep <- df[-grep("\\.y",colnames(df))]
colnames(df_del_rep) <- sub("\\.x", "", colnames(df_del_rep))

#count missing proportions
NA_rate <- c()
for (i in 1:ncol(df_del_rep)){
  NA_rate <- c(NA_rate, round(sum(is.na(df_del_rep[i]))/nrow(df_del_rep[i]),4)*100)
}
NA_rates <- data.frame(colnames(df_del_rep),NA_rate)

#Classifier the Data Set
class_var = c('direction','class','draw_advantage','all_weather',
              'last_winner_bred','official_rating_type','speed_rating_type',
              'private_handicap_type','last_race_type_id','last_race_type',
              'last_race_beaten_fav','days_since_ran_type')

num_var = c('rating','official_rating','speed_rating','prize_pos_4',
            'last_winner_runners','last_winner_age','last_winner_weight',
            'last_winner_sp_decimal','distance_travelled','official_rating.1',
            'forecast_price_decimal','distance_beaten','distance_behind_winner',
            'days_since_ran','adjusted_rating','dam_year_born','sire_year_born',
            'dam_sire_year_born')

unknown_var = c('last_winner_runner_id','last_winner_name','last_winner_trainer',
                'last_winner_trainer_id','last_winner_jockey','last_winner_jockey_id',
                'last_winner_sp','form_figures','stall_number','betting_text')

#Missing value interpolation
df_MissDelete_1 <- df_del_rep[,-which(NA_rates$NA_rate >= 60)]
df_MissDelete_2 <- df_del_rep[,which(NA_rates$NA_rate <= 1)]
df_MissDelete_2 <- df_MissDelete_1[complete.cases(df_MissDelete_2),]
df_MissDelete_8 <- df_MissDelete_2[which(!is.na(df_MissDelete_2$finish_position)),]

for (i in class_var){
  quan <- table(df_MissDelete_8[i])
  pro <- quan/sum(quan)
  impute_value <- sample(names(quan),length(which(is.na(df_MissDelete_8[i]))),
                         replace = TRUE,prob = pro)
  df_MissDelete_8[i][is.na(df_MissDelete_8[i])] <- impute_value
}

for (i in num_var){
  impute_value <- round(mean(df_MissDelete_8[i][!is.na(df_MissDelete_8[i])]),2)
  df_MissDelete_8[i][is.na(df_MissDelete_8[i])] <- impute_value
}

df_MissDelete_6 <- df_MissDelete_8[,-which(colnames(df_MissDelete_8) %in% unknown_var)]
df_MissDelete_7 <- na.omit(df_MissDelete_6)

#save as csv file
write.csv(df_MissDelete_7,"C:\\Users\\Eric\\Desktop\\df_preprocessed.csv")


