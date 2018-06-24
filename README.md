# Protagonist-Antagonist-Classifier
A DNN that uses UBCG features to classify noun phrases into one of two named entity classes: Protagonist or Antagonist. Training and Validation data are from the Gutenberg corpus.

IMPORTANT NOTE:

The following project was created during a recent interview. The basic premise behind the features used in this system
is that the semantic role of a named entity is limited to one of only a few possible role assignments, which are
accessible to speakers through syntactic templates known as constructions, per Goldberg 2006, and validated in part
by research done by Reisinger et al 2016, which indicated that syntactic roles carry with them certain semantic 
features of agentivity and patientivity, a priori.

Thus, the system makes the following assumption: Given a sentence like 

"Starbuck hauled the brooms to port."

We can infer from their syntactic roles the following semantic role assignments.

AGENT: Starbuck
PATIENT: the brooms
LOCATION/RECIPIENT: port

To represent these relationships, we can create an embedding for tuples of the verb and nominal argument, as in
("Starbuck", "hauled"), and we assume that the embedding for this tuple would be sufficiently similar to other
instantiations of the AGENT role as to classify them similarly. The process is the same for all roles in the
utterance.

Part of the task was simultaneously to capture whether or not the named entity was the protagonist or antagonist of 
the story. The assumption here is similar, such that protagonists are likely to have verbs/nominal sisters of 
positive polarity, and antagonists with negative. Compare

("Starbuck", "hauled")

to

("Cthulhu", "lurked")

F1 for validation data was found to be 1.0, however in the test data the system appeared to capture some non-named
entities as well that were in fact in negatively polar semantic roles.
