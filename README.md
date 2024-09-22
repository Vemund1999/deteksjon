# deteksjon
make sure to be on the master branch, at https://github.com/Vemund1999/deteksjon/tree/master

the .pt models are not included in the project for now due to big file sizes.

The project tracks:
- cars driving on the road
- the color on the car, using k-means clustering
- an ID is added to the car
- counts of the cars, how many cars are being spotted, and how many have been spotted

the project also creates a coordinate system for the cars.


# how to run
there is currently no model included in the project.
first such a model has to be trained by running all the cells in
```
training/car_and_carplate_detection.ipynb
```
then the best.pt model has to be refrenced in the main.py file at
```
tracker = Tracker('best.pt')
```


then run the file
```
python main.py
```

![bilde](https://github.com/user-attachments/assets/4f7ba484-6355-4b4f-80ec-f2fb206bc29a)








