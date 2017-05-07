import csv

from get_data import get_data

def write_csv():
    ratings = get_data()
    with open('rating_matrix.csv', 'w+') as fp:
        writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['UserID'] + list(range(ratings.shape[1])))
        for i,rating in enumerate(ratings):
            #Write item to outcsv
            writer.writerow([i] + rating.tolist())

if __name__ == '__main__':
    write_csv()