# classification with stripe layers
A high order approach to image classification.  Uses the "stripe" layers for "convolutional" and
fully connected layers.  However, neither of these are the same as the standard version.  Instead
of convolutional blocks, stripes across the entire domain are used along with a positional encoding.
The "fully connected" layer still shares weights along each stripe, but the output of that layer takes
the inputs from all of the stripes (and their rotations).

##
Convolutional like
```
python cifar100.py segments=10 n=5 max_epochs=10 rotations=2 periodicity=2.0
```
##
Fully connected like
```
python cifar100invariant.py segments=10 n=3 max_epochs=10 rotations=1 periodicity=2.0 out_features=50
```
