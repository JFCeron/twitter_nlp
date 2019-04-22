Sys.setlocale("LC_ALL","spanish")
tweets = read.csv("tweets_venezuela.csv", stringsAsFactors=F, encoding="latin1")
write.csv(data.frame(unique(tweets$CONTENT)),"tweets_unicos.csv",row.names=F)