Understanding of how should the system be designed. Identifies the main components and its relationships. 
# Agility and architecture
In early stage of agile processes is designed the overall system architecture. Refactoring is usually expensive as it affects many components.
# Abstraction
Concerned about how an individual program is decomposed into components.
# Advantages
- Stakeholder benefits from discussion about the architecture.
- Analysis if achieving the requirements is possible.
- May be reusable across a range of systems.
# Architectural representations
You reuse the design from something that had already been done. Are simple informal block diagrams showing entities and relationships, criticized for the lack of semantics and not show the types of relationships nor the visible properties. 
# Use of architectural models
Is an abstract model which allows to oversight small details while keeping up with the overall system. Stakeholders can relate to it and understand it. As a way of documenting its components and its relationships.
# Architectural design decisions
Process differs from the type of system being developed. Modelling systems, transactions, systems of systems, entertainment, etc. Common decisions may span all design processes affecting non functional requirements of the system.
# Architecture reuse
Systems in the same domain reflect domain concepts. Product lines are built around a core architecture satisfying a particular customer. Can be instantiated in different ways.
# System characteristics
- Performance, critical operations and minimise communications. Use large than fine-grained components.
- Security, layered architecture critical assets in the inner layers.
- Safety, localize security critical in a small set of sub-systems.
- Availability, include redundant components.
- Maintenance, fine-grain easily replaceable components.
# Architectural models
Each shows only one perspective.
- Logical view, abstraction as objects or objects classes.
- Process view, at run-time is composed of interacting processes.
- Development view, how is decomposed for development.
- Physical view, hardware components and how are distributed in the system.
- Relating use cases or scenarios.
# Representing
UML is an appropriate notation. May differ as there are no abstraction for high level ones.
# Architectural patterns
Represents, shares and reuses knowledge. Stylized description of good design. Should include info on when it's useful and when it's not. Represented with graphical or tabular.
# MVC (Modelo Vista Controlador)
*Read all Wikipedia of Design Patterns!*
*Read all GRASP*
# Layered architectures
organises the system in a set of layers.
Supports incremental development of sub-systems in different layers.
Artificial to structure a system in such a way.
## Repository architecture
Subsystems must share data.
- Share data is held in central database.
- Each subsystem hold its own database and share explicitly.
# Client server architecture
