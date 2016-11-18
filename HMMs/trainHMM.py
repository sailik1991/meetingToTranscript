from hmmlearn.hmm import MultinomialHMM

M = 2

if __name__ == "__main__":
    
    # Initialize and train an HMM model
    model = MultimonialHMM( n_components = M, n_iter = 100 )
    
