Developing abstract models of a system with a different view or perspective. Only focusing on the important things. Representing the system using some graphical notation most notably UML. Models are used to communicate better with the system.

# Existing and planned system models
Existing system are used during requirements engineering, which can lead to new requirements. 

System perspectives, are external about the environment, interaction between the system and its environment between its components, structural data organization processed by the system and behavioral dynamism of the system on its response to events.

# Graphical models
## UML diagram types
- Activity diagrams, activities involved in a process or data processing.
- Use case diagrams, interactions between system and its environment.
- Sequence diagrams, interactions between actors and the system and its components.
- Class diagrams, shows object classes and associations between them.
- State diagrams, reaction to internal and external events.

## Use of graphical models
Incomplete and incorrect models are OK because they support discussion, as it is always in constant evolution. As a way of documenting the system, precise but not necessarily complete. Detailed description that can be used to generate s system implementation.

# Context models
Illustrate the operation context of the model, showing what lies outside of the system boundaries. Social and organisational concerns affect decision on where to put such boundaries. Architectural models show the system and its relationship with others.
## System boundaries
Define what is or not in the system. Showing which systems are used or the ones that depend on the system being developed. Boundaries may be set in a political way, pressure to increase or decrease the influence in different parts of the organization.

# Interaction models
Help to identify user requirements, system-to-system highlights possible communication problems. Helps understand if a proposed system structure is likely to deliver system performance and dependability. Use case and sequence diagrams, may be used.

# Structural models
Software organization of system in terms of its components that make up that system and their relationships. May be static models showing the structure of the system being designed. Or dynamic while it's being executed. Used when discussing and designing the system architecture. Class diagrams are part of it.

# Behavioral models
For the dynamical behavior of a system while executing. How the system responds to environment stimuli. As in data that arrives and has to be processed or as event that trigger event processing.
## Data-driven modeling
Controlled by data input to the system, with relatively little vent processing. Show the sequence of actions involved in processing input data and generating an associated output. Useful in analysis as can show end-to-end processing in a system.
## Event-driven modeling
Not a lot of data. Shows how a system responds to external and internal events. System has a finite number of states and events that may cause a transition from one to another.
## State machine models
Response to stimuli, may be used for real time events. States as nodes and events as arrows. Statecharts UML may be used to represent them.

# Model-driven engineering
Based on models rather than programs. Programs generated from the platform. Raises level of abstraction, so programmer don't have to be concerned with language or evolution of the platform. Cheaper to adapt systems, but must be taken into consideration that different programs may increase the cost for utilities such as translators and such.

# Model-driven architecture
Focused on software must be developed by a subset of UML models to generate the system. From high-level, platform independent, no manual intervention.
## Types of model
- Computation independent model, abstractions used in a system. Aka domain models.
- Platform independent model, no reference to implementation. Show system interaction with UML.
- Platform specific model, implementation of required specific platform details.

## Agile methods and MDA
Intended to support iterative approaches being able to be used within agile methods. 
### Adoption of MDA
Limits un usage of MDA, specialized support from one level to another. Limited tools that are available. Companies are reluctant to develop their own tools or rely in small ones that may go out of business. Abstraction may not be the right ones for abstractions. Implementation is not the major problem, requirements, security and dependability, integration with legacy systems and testing  are more important. Arguments for platform independence are only valid for large, long-lifetime systems.