def plot_the_line(df_x,df_y,svm):
    h = .02     

    x_min, x_max = df_x['PCA1'].min() - 1, df_x['PCA1'].max() + 1
    y_min, y_max = df_x['PCA2'].min() - 1, df_x['PCA2'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    
    x_min, x_max = df_x['PCA1'].min() - 1, df_x['PCA1'].max() + 1
    y_min, y_max = df_x['PCA2'].min() - 1, df_x['PCA2'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z= svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(df_x['PCA1'], df_x['PCA2'], c=df_y.values, alpha=0.8)
    plt.xlabel("PCA 1",fontsize=15)
    plt.ylabel("PCA 2",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
