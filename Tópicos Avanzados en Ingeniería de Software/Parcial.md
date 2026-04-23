# Chapter 5
Representation of a system can be done graphically with UML and a formal mathematical model. Describes the system to the engineers developing it and serves as documentation.

1. Models are used during requirement engineering with existing systems serve to clarify and explain to stakeholders.
2. Also in new systems to discuss design proposals and document system for implementation.

A model is an abstract representation of the system.

## Four development perspectives
1. External, models the environment.
2. Interaction, between a system and it's environment.
3. Structural, organization of the system or data processed by the system.
4. Behavioral, dynamic of how the system responds to events.

## Graphical models
### Common uses
1. Stimulate discussion of an existing or proposes system. Models might be incomplete, notation might be used informally. Used in agile.
2. Document an existing system. Do not need to be complete as can describe only a part. Correct notation and accurate description of the system.
3. Detailed system description. Complete, used in model-driven development (source code is generated from model). 

### UML Main models
1. **Activity diagrams**, show activities in process or data processing. ![[Pasted image 20260422182238.png]]
2. **Use case diagrams**, interaction between system and its environment.![[Pasted image 20260422162543.png]]
3. **Sequence diagrams**, show interactions between actors and the system. Also between system components. ![[Pasted image 20260422180122.png]]
4. **Class diagrams**, OOP. ![[Pasted image 20260422181234.png]]
5. **State diagrams**, how does the system reacts to internal and external events. ![[Pasted image 20260422182113.png]]

## Context Models
In the beginning you should define the scope, and system boundaries according to user requirements and needs.
Then define the context and the system dependencies.
A context model do not show the type of relationships between the systems. 
![[Pasted image 20260422184107.png]]
Activity diagrams, may be used too.

## Interaction Models
**Use case diagrams**, interactions between a system and external agents (actores). 
**Sequence diagrams**, interactions between actor with objects and between system components (objects themselves).
*Comunication diagrams*, similar to sequence ones.

Tabular data, remember Guillermito's instructions.

A single *use case* might be described as what a user expects from a system. **Use case** diagrams are good in early stages of development, not so much in requirements engineering. 

Sequence of interactions during a particular use case or use case instance. 

## Structural Models
May be static, showing the organization of a system designed. 
May be dynamic, showing the organization of a system being executed.

Used while designing and discussing the system architecture. For the whole system or the objects in the system and their relationships.

**Class diagrams**, OOP. Show classes in system and its associations with others.

