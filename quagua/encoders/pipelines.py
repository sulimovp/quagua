"""
Pipeline classes for combining encoders and mixers
"""


class EncodeThenMixPipeline:
    """
    Pipeline: First encode (irreversible transformation), then mix (feature combination)
    This implements the two-stage approach: Encode → Mix
    """
    
    def __init__(self, encoder, mixer):
        """
        encoder: Irreversible encoding method (e.g., RandomOneToManyEncoder, ChaoticSystemEncoder)
        mixer: Mixing method (e.g., ClassicalXORMixer, QuantumEntanglementEncoder)
        """
        self.encoder = encoder
        self.mixer = mixer
        
    def fit(self, X):
        """Fit encoder, then fit mixer on encoded data"""
        # Step 1: Fit encoder
        self.encoder.fit(X)
        X_encoded = self.encoder.transform(X)
        
        # Step 2: Fit mixer on encoded data
        self.mixer.fit(X_encoded)
        
        return self
    
    def transform(self, X):
        """Encode, then mix"""
        # Step 1: Encode
        X_encoded = self.encoder.transform(X)
        
        # Step 2: Mix
        X_mixed = self.mixer.transform(X_encoded)
        
        return X_mixed
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



