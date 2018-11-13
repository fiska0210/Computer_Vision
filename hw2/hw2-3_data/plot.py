import matplotlib.pyplot as plt


loss = [0.3418, 0.1212, 0.0905, 0.075, 0.06, 0.0597, 0.0541, 0.0495, 0.0471, 0.0436, 0.0416, 0.0387]
acc = [ 0.8969, 0.9649, 0.9729, 0.9772, 0.9811, 0.9814, 0.9836, 0.9855, 0.986, 0.987, 0.9878, 0.9885]
plt.plot(loss)

print(len(loss))
print(len(acc))
plt.subplots_adjust(wspace = 0.5)
plt.subplot(1,2,1)
plt.ylabel("Training Loss") 
plt.ylabel("Training Accuracy") 
plt.title("Learning Curve")
plt.plot(acc) 
plt.subplot(1,2,2)
plt.xlabel("Iteration")
plt.ylabel("Training Loss") 
plt.title("Learning Curve")
plt.plot(loss) 
plt.savefig("train.png",dpi=300,format="png") 
plt.show()
plt.pause(5)
plt.close()

# loss: 0.3418  acc: 0.8969
# loss: 0.1212  acc: 0.9649
# loss: 0.0905  acc: 0.9729
# loss: 0.0750  acc: 0.9772
# loss: 0.0655  acc: 0.9811
# loss: 0.0597  acc: 0.9814
# loss: 0.0541  acc: 0.9836
# loss: 0.0495  acc: 0.9855
# loss: 0.0471  acc: 0.9860
# loss: 0.0436  acc: 0.9870
# loss: 0.0416  acc: 0.9878
# loss: 0.0387  acc: 0.9885

# Test loss: 0.030516302801910207
# Test accuracy: 0.9899
