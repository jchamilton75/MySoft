



nn = 512
x= np.linspace(-1,1,nn)
xx, yy = np.meshgrid(x,x)

ax = 3.
ay=3.
ripples = np.sin(ax*np.pi*xx + ay*np.pi*yy+np.pi/2)
clf()
imshow(ripples)


gsig = 0.2
gauss = np.exp(-0.5 * (xx**2+yy**2)/gsig**2)
clf()
imshow(gauss)

clf()
imshow(ripples**2*gauss)

clf()
imshow(ripples**2*gauss,cmap=matplotlib.cm.get_cmap('gist_ncar'))
