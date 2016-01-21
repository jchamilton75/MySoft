
### par millimetre ou degre
tm1 = np.array([0.03, 0.04, 0.05])
rm1 = np.array([0.07, 0.08, 0.12])
tm2 = np.array([0.06, 0.06, 0.11])
rm2 = np.array([0.11, 0.10, 0.15])

tolt = np.linspace(0,1,90)
tolr = np.linspace(0,1,100)

eff = np.zeros((90,100))
for i in xrange(90):
        for j in xrange(100):
            eff[i,j] = np.sqrt(np.sum((tm1*tolt[i])**2) + np.sum((tm2*tolt[i])**2) + np.sum((rm1*tolr[j])**2) + np.sum((rm2*tolr[j])**2))
            
clf()            
imshow(eff, origin='lower', extent=[0.,1.,0.,1.])
colorbar()    
contour(tolr, tolt, eff, levels=[0.1])
xlabel('Rotation in deg.')
ylabel('Translation in mm')           

dim= 200.
depl = dim * np.sin(np.radians(tolr))
plot(tolr, depl, color='red', lw=3,label='Expected translation for 200mm size')
xlim(0,1)
ylim(0,1)
legend()
savefig('tolerances.png')






