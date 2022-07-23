import glob

import pandas

class preprocessing():

    def preprocessing(self):
        self.count = {'sport': 0, 'world': 0, "us": 0, "business": 0, "health": 0, "entertainment": 0, "sci_tech": 0}

        with open('news.news', 'r') as f:
            text = f.read()
            news = text.split("\n\n")
            count = {'sport': 0, 'world': 0, "us": 0, "business": 0, "health": 0, "entertainment": 0, "sci_tech": 0}
            for news_item in news:
                lines = news_item.split("\n")
                if(len(lines)==7):
                    file_to_write = open('data/' + lines[6] + '/' + str(count[lines[6]]) + '.txt', 'w+',encoding='utf8')
                    count[lines[6]] = count[lines[6]] + 1
                    file_to_write.write(news_item)  # python will convert \n to os.linesep
                    file_to_write.close()

        print(self.count)

        self.category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
        self.directory_list = ["data/sport/*.txt", "data/world/*.txt", "data/us/*.txt", "data/business/*.txt", "data/health/*.txt",
                          "data/entertainment/*.txt", "data/sci_tech/*.txt", ]

        text_files = list(map(lambda x: glob.glob(x), self.directory_list))
        text_files = [item for sublist in text_files for item in sublist]

        print(text_files)
        training_data = []

        for t in text_files:
            f = open(t, 'r')
            f = f.read()
            t = f.split('\n')
            training_data.append({'data': t[0] + ' ' + t[1], 'flag': self.category_list.index(t[6])})

        training_data = pandas.DataFrame(training_data, columns=['data', 'flag'])
        training_data.to_csv("train_data.csv", sep=',', encoding='utf-8')
