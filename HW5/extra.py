eigvals, pcomps = pca(x, n_comps=748)
err_means, err_pcomps = np.zeros(748), np.zeros(748)
for n in range(748):
    em, ep = calc_errs(x, pcomps[:n])
    err_means[n] = em
    err_pcomps[n] = ep

plt.plot(np.arange(748), err_means)
plt.plot(np.arange(748), err_pcomps)
plt.show()