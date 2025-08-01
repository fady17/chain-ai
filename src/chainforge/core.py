"""
Core Abstract Interfaces for the ChainForge Framework.

This module defines the fundamental, generic `Runnable` interface that is the
cornerstone of the entire framework. It prioritizes type safety, flexibility,
and composability, allowing developers to chain together components that operate
on any Python objects.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Awaitable, Callable, Generic, List, TypeVar

# --- Generic Type Variables ---
# These TypeVars allow us to create strongly-typed Runnables. A Runnable is a
# transformation from an InputT to an OutputT. This is a much more flexible
# and type-safe approach than enforcing a specific Pydantic base class.
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Runnable(Generic[InputT, OutputT], ABC):
    """
    An abstract base class for a composable, asynchronous, generic component.

    This interface is the heart of ChainForge. It defines a contract for any
    piece of logic that can be "run" as part of a sequence. By using generics,
    we ensure that the data flowing between components in a chain can be
    statically type-checked, drastically reducing runtime errors.
    """

    @abstractmethod
    async def ainvoke(self, input: InputT, **kwargs: Any) -> OutputT:
        """Asynchronously invokes the component with a single input."""
        raise NotImplementedError

    async def abatch(self, inputs: List[InputT], **kwargs: Any) -> List[OutputT]:
        """
        Asynchronously invokes the component on a batch of inputs in parallel.

        The default implementation uses `asyncio.gather` for simple concurrency.
        Components that can leverage more efficient, provider-specific batching
        (e.g., a single API call for multiple documents) should override this.
        """
        return await asyncio.gather(*(self.ainvoke(i, **kwargs) for i in inputs))

    @abstractmethod
    async def astream(self, input: InputT, **kwargs: Any) -> AsyncIterator[OutputT]:
        """Asynchronously streams the output of the component."""
        # This is an async generator. The `if False` construct is a common way
        # to make an abstract method an async generator without executing code.
        if False:
            yield

    def __or__(self, other: "Runnable[OutputT, Any]") -> "RunnableSequence":
        """
        Enables chaining of `Runnable` components using the `|` operator.

        This creates a `RunnableSequence`. The type hinting `Runnable[OutputT, Any]`
        helps enforce that the next item in the chain can accept the output of
        the current item.

        Returns:
            A `RunnableSequence` that encapsulates the two components.
        """
        return RunnableSequence(first=self, second=other)


class RunnableSequence(Runnable[InputT, OutputT]):
    """
    A `Runnable` that composes two other `Runnables` in a sequence.
    """

    def __init__(self, first: Runnable[InputT, Any], second: Runnable[Any, OutputT]):
        """
        Initializes the sequence. The components are stored internally.
        """
        self.first = first
        self.second = second

    async def ainvoke(self, input: InputT, **kwargs: Any) -> OutputT:
        """

        Executes the sequence: `second.ainvoke(await first.ainvoke(input))`.
        """
        # Await the result of the first runnable...
        first_output = await self.first.ainvoke(input, **kwargs)
        # ...and pass it as input to the second runnable.
        return await self.second.ainvoke(first_output, **kwargs)

    async def astream(self, input: InputT, **kwargs: Any) -> AsyncIterator[OutputT]:
        """
        Streams the output of the sequence.

        This implementation invokes the first part of the chain and then streams
        the output of the second part. This is the standard model for streaming
        in a sequence, as intermediate steps are typically not streamable.
        """
        first_output = await self.first.ainvoke(input, **kwargs)
        # The `async for` construct correctly handles the async iterator protocol.
        async for chunk in self.second.astream(first_output, **kwargs):
            yield chunk

    def __or__(self, other: "Runnable[OutputT, Any]") -> "RunnableSequence":
        """
        Extends the sequence by creating a new, longer sequence.
        The existing sequence becomes the "first" part of the new one.
        """
        return RunnableSequence(first=self, second=other)


class RunnableLambda(Runnable[InputT, OutputT]):
    """

    A `Runnable` that wraps an arbitrary async callable (function or lambda).

    This is a powerful utility component that allows developers to easily insert
    custom, non-`Runnable` logic into a chain without the boilerplate of creating
    a new class.
    """
    def __init__(self, func: Callable[[InputT], Awaitable[OutputT]]):
        """
        Args:
            func: The async function to wrap. It must take one argument and
                  return an awaitable (the result).
        """
        self.func = func

    async def ainvoke(self, input: InputT, **kwargs: Any) -> OutputT:
        """Invokes the wrapped async function."""
        return await self.func(input)

    async def astream(self, input: InputT, **kwargs: Any) -> AsyncIterator[OutputT]:
        """
        Fulfills the stream contract by invoking the function and yielding
        the single result. This assumes the wrapped function is not a generator.
        """
        result = await self.ainvoke(input, **kwargs)
        yield result